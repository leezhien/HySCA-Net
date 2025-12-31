import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from sklearn.decomposition import PCA
from operator import truediv
import time
import sys
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import pandas as pd

from dataAug3 import augment_hyperspectral_data1
from model import FullModel1
from splitTrainTestSet import splitTrainTestSet1

zh_font_path = r"C:\Windows\Fonts\simhei.ttf"
if os.path.exists(zh_font_path):
    zh_font = font_manager.FontProperties(fname=zh_font_path)
    plt.rcParams["font.family"] = zh_font.get_name()
    font_manager.fontManager.addfont(zh_font_path)

plt.rcParams['axes.unicode_minus'] = False


class EarlyStopping:
    def __init__(self, patience=30, verbose=False, delta=0.0001, path='checkpoint.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

//加载数据集
def loadData():
    data = sio.loadmat(r'..\datasets\PaviaU\PaviaU.mat')['paviaU']
    labels = sio.loadmat(r'..\datasets\PaviaU\PaviaU_gt.mat')['paviaU_gt']
    return data, labels


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.memmap(r'E:\\data.memmap', dtype='float32', mode='w+',
                            shape=(X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchesIndices = []

    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchesIndices.append((r - margin, c - margin))
            patchIndex = patchIndex + 1

    if removeZeroLabels:
        validIndices = patchesLabels > 0
        patchesData = patchesData[validIndices, :, :, :]
        patchesLabels = patchesLabels[validIndices]
        patchesLabels -= 1
        patchesIndices = np.array(patchesIndices)[validIndices]
    return patchesData, patchesLabels, patchesIndices


class TrainDS(torch.utils.data.Dataset):
    def __init__(self, Xtrain, ytrain):
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)
        self.len = Xtrain.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class ValDS(torch.utils.data.Dataset):
    def __init__(self, Xval, yval):
        self.x_data = torch.FloatTensor(Xval)
        self.y_data = torch.LongTensor(yval)
        self.len = Xval.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class TestDS(torch.utils.data.Dataset):
    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def train_model(model, device, dataloader, optimizer, loss_fn, epoch, epochs, scheduler=None):
    model.train()
    loss, accurate, sample_num = 0.0, 0.0, 0
    tqdm_dataloader = tqdm(dataloader, file=sys.stdout)
    for batch_index, (data, target) in enumerate(tqdm_dataloader):
        sample_num += data.shape[0]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        features, output = model(data)
        cur_loss = loss_fn(output, target)
        cur_loss.backward()
        pred = output.argmax(dim=1)
        cur_acc = pred.eq(target.data.view_as(pred)).sum()

        optimizer.step()
        loss += cur_loss.item()
        accurate += cur_acc.item()
        current_lr = optimizer.param_groups[0]["lr"]

        tqdm_dataloader.desc = "[train epoch {}/{}] loss: {:.4f}, lr: {:.5f}".format(
            epoch + 1, epochs, loss / sample_num, current_lr)

    train_acc = accurate / len(dataloader.dataset)
    avg_loss = loss / len(dataloader.dataset)
    return avg_loss, train_acc * 100


def validate_model(model, device, dataloader, loss_fn, epoch, epochs):
    model.eval()
    loss, accurate, sample_num = 0.0, 0.0, 0
    tqdm_dataloader = tqdm(dataloader, file=sys.stdout)
    with torch.no_grad():
        for batch_index, (data, target) in enumerate(tqdm_dataloader):
            sample_num += data.shape[0]
            data, target = data.to(device), target.to(device)
            features, output = model(data)
            cur_loss = loss_fn(output, target)
            pred = output.argmax(dim=1)
            cur_acc = pred.eq(target.data.view_as(pred)).sum()

            loss += cur_loss.item()
            accurate += cur_acc.item()

            tqdm_dataloader.desc = "[validate epoch {}/{}] loss: {:.4f}".format(
                epoch + 1, epochs, loss / sample_num)

    val_acc = accurate / len(dataloader.dataset)
    avg_loss = loss / len(dataloader.dataset)
    return avg_loss, val_acc * 100


def test_model(savepath, model, device, dataloader, loss_fn, class_name):
    model.eval()
    count, loss, accurate = 0, 0.0, 0.0
    features_list = []
    prob_matrix_list = []

    with torch.no_grad():
        tqdm_dataloader = tqdm(dataloader, file=sys.stdout)
        infer_start = time.perf_counter()
        for batch_index, (data, target) in enumerate(tqdm_dataloader):
            data, target = data.to(device), target.to(device)
            features, output = model(data)
            features_list.append(features.cpu())
            prob_matrix = torch.softmax(output, dim=1).cpu().numpy()
            prob_matrix_list.append(prob_matrix)

            cur_loss = loss_fn(output, target)
            pred = output.argmax(dim=1)
            cur_acc = pred.eq(target.data.view_as(pred)).sum()
            outputs = np.argmax(output.detach().cpu().numpy(), axis=1)
            target = target.cpu()
            if count == 0:
                y_pred = outputs
                y_test = target
                count = 1
            else:
                y_pred = np.concatenate((y_pred, outputs))
                y_test = np.concatenate((y_test, target))
            loss += cur_loss.item()
            accurate += cur_acc.item()

        infer_end = time.perf_counter()

        features_all = torch.cat(features_list, dim=0).numpy()
        prob_matrix_all = np.vstack(prob_matrix_list)
        y_test = np.array(y_test)

        test_acc = accurate / len(dataloader.dataset)
        loss /= len(dataloader.dataset)
        print(f"test_loss : {loss}")
        print(f"test_acc : {100 * test_acc}")
        print(f"Total Inference Time (without data loading): {infer_end - infer_start:.4f}s")

        classification = classification_report(y_test, y_pred, target_names=class_name)
        oa = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)

        each_acc, aa = AA_andEachClassAccuracy(confusion)
        kappa = cohen_kappa_score(y_test, y_pred)
        print('oa:', oa)
        print('aa:', aa)
        print('each_acc:', each_acc)
        print('kappa :', kappa)

        with open(file_path1, 'a') as t_f:
            t_f.write(f'Overall accuracy:{oa * 100}' + "\n")
            t_f.write(f'Average accuracy:{aa * 100}' + "\n")
            t_f.write(f'Kappa accuracy:{kappa * 100}' + "\n")
            t_f.write(f'Each accuracy:{each_acc * 100}' + "\n")
    return oa, aa, kappa, each_acc


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


if __name__ == '__main__':

    X, y = loadData()
    class_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees',
                   'Painted metal sheets', 'Bare soil', 'Bitumen',
                   'Self-Blocking Bricks', 'Shadows']
    patch_size = 15
    X, y, patchesIndices = createImageCubes(X, y, windowSize=patch_size)

    x_train, x_val, x_test, y_train, y_val, y_test, train_indices, val_indices, test_indices = splitTrainTestSet1(
        X, y, patchesIndices)

    Xtrain = x_train.reshape(-1, patch_size, patch_size, 103)
    Xtest = x_test.reshape(-1, patch_size, patch_size, 103)
    Xval = x_val.reshape(-1, patch_size, patch_size, 103)
    Xtrain = Xtrain.transpose(0, 3, 1, 2)
    Xtest = Xtest.transpose(0, 3, 1, 2)
    Xval = Xval.transpose(0, 3, 1, 2)

    res_aug = augment_hyperspectral_data1(Xtrain, y_train, neighborhood_size=2)
    if len(res_aug) == 3:
        X_train_augmented, y_train_augmented, _ = res_aug
    else:
        X_train_augmented, y_train_augmented = res_aug

    X_train = np.concatenate((Xtrain, X_train_augmented), axis=0)
    y_train = np.concatenate((y_train, y_train_augmented), axis=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    unique_classes, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    num_classes = len(unique_classes)
    class_weights = total_samples / (num_classes * counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    trainset = TrainDS(X_train, y_train)
    print("训练集的长度为:", len(trainset))
    testset = TestDS(Xtest, y_test)
    valset = ValDS(Xval, y_val)
    train_loader = DataLoader(dataset=trainset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=valset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=testset, batch_size=32, shuffle=False, num_workers=0)

    save_path = "results"
    temp_save_dir = os.path.join(os.path.dirname(__file__), "tmp_results")
    if os.path.exists(temp_save_dir):
        save_path = temp_save_dir
    else:
        os.makedirs(save_path, exist_ok=True)

    train_message = f"{save_path}/result.txt"
    file_path1 = f"{save_path}/res.txt"
    results_file_path = f"{save_path}/results_summary.txt"

    epochs = 500
    total_rounds = 5
    time_11 = 1

    patience_val = 30
    delta_val = 0.0001

    OA_ALL, AA_ALL, KPP_ALL, EACH_ACC_ALL = [], [], [], []

    train_start = time.time()

    for i in range(time_11):
        for r in range(total_rounds):
            model = FullModel1().to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            optimizer = optim.Adam(model.parameters(), lr=0.00005, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=15,
                verbose=True,
                min_lr=1e-7
            )

            model_save_path = f"{save_path}/best_model_weight.pth"
            early_stopping = EarlyStopping(patience=patience_val, verbose=True, delta=delta_val, path=model_save_path)

            all_train_losses = []
            all_val_losses = []
            all_train_accuracies = []
            all_val_accuracies = []

            print(f"\n--- Round {r + 1} Training Starts ---")

            best_epoch_record = 0

            for epoch in range(epochs):
                train_loss, train_accuracy = train_model(model, device, train_loader, optimizer, criterion, epoch,
                                                         epochs)
                val_loss, val_accuracy = validate_model(model, device, val_loader, criterion, epoch, epochs)

                scheduler.step(val_loss)

                all_val_losses.append(val_loss)
                all_val_accuracies.append(val_accuracy)
                all_train_losses.append(train_loss)
                all_train_accuracies.append(train_accuracy)

                early_stopping(val_loss, model)

                if early_stopping.counter == 0:
                    best_epoch_record = epoch

                if early_stopping.early_stop:
                    print(f"Early stopping triggered at Epoch {epoch + 1}! Best Loss: {early_stopping.best_score:.6f}")
                    with open(train_message, 'w') as t_f:
                        t_f.write(
                            f"Early Stop at {epoch + 1} epochs, best_val_loss: {-early_stopping.best_score:.6f}" + "\n")
                    break

            train_end = time.time()
            train_time = train_end - train_start
            time_train_now = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(train_message, 'a') as t_f:
                t_f.write(f"NOW:{time_train_now}" + "\n")
                t_f.write(f"Round {r + 1} finished. Duration: {train_time / 60:.2f}min!" + "\n")

            model.load_state_dict(torch.load(model_save_path, map_location=device))

            criterion_test = nn.CrossEntropyLoss()
            oa, aa, kappa, each_acc = test_model(save_path, model, device, test_loader, criterion_test, class_names)
            OA_ALL.append(oa)
            AA_ALL.append(aa)
            KPP_ALL.append(kappa)
            EACH_ACC_ALL.append(each_acc)

        print('List of OA:', list(OA_ALL))
        print('List of AA:', list(AA_ALL))
        print('List of KPP:', list(KPP_ALL))
        print('OA=', round(np.mean(OA_ALL) * 100, 2), '+-', round(np.std(OA_ALL) * 100, 2))
        print('AA=', round(np.mean(AA_ALL) * 100, 2), '+-', round(np.std(AA_ALL) * 100, 2))
        print('Kpp=', round(np.mean(KPP_ALL) * 100, 2), '+-', round(np.std(KPP_ALL) * 100, 2))
        print('Acc per class=', np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2), '+-',
              np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2))

        with open(results_file_path, 'a') as results_file:
            results_file.write('List of OA: ' + str(list(OA_ALL)) + '\n')
            results_file.write('List of AA: ' + str(list(AA_ALL)) + '\n')
            results_file.write('List of KPP: ' + str(list(KPP_ALL)) + '\n')
            results_file.write(
                'OA= ' + str(round(np.mean(OA_ALL) * 100, 2)) + ' ± ' + str(round(np.std(OA_ALL) * 100, 2)) + '\n')
            results_file.write(
                'AA= ' + str(round(np.mean(AA_ALL) * 100, 2)) + ' ± ' + str(round(np.std(AA_ALL) * 100, 2)) + '\n')
            results_file.write(
                'Kpp= ' + str(round(np.mean(KPP_ALL) * 100, 2)) + ' ± ' + str(round(np.std(KPP_ALL) * 100, 2)) + '\n')
            results_file.write('Acc per class= ' + str(np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2)) + ' ± ' +
                               str(np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2)) + '\n')

            OA_ALL = []
            AA_ALL = []
            KPP_ALL = []
            EACH_ACC_ALL = []