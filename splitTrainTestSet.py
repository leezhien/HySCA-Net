import numpy as np
import pandas as pd

def splitTrainTestSet1(X, y, patchesIndices, num_per_class=10, num_val_per_class=5):
    num_samples, dim1, dim2, dim3 = X.shape
    X_flatten = X.reshape(num_samples, dim1 * dim2 * dim3)
    data = pd.DataFrame(X_flatten)
    data['label'] = y
    data['patch_index'] = list(patchesIndices)
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()
    print(np.unique(y))
    for label in np.unique(y):
        class_data = data[data['label'] == label]
        if len(class_data) <= num_per_class + num_val_per_class:
            train_data = pd.concat([train_data, class_data])
        else:
            class_train_data = class_data.sample(n=num_per_class, random_state=345)
            remaining_data = class_data.drop(class_train_data.index)
            class_val_data = remaining_data.sample(n=num_val_per_class, random_state=345)
            class_test_data = remaining_data.drop(class_val_data.index)

            train_data = pd.concat([train_data, class_train_data])
            val_data = pd.concat([val_data, class_val_data])
            test_data = pd.concat([test_data, class_test_data])
    X_train = train_data.drop(columns=['label', 'patch_index']).values
    y_train = train_data['label'].values
    train_indices = train_data['patch_index'].values

    X_val = val_data.drop(columns=['label', 'patch_index']).values
    y_val = val_data['label'].values
    val_indices = val_data['patch_index'].values

    X_test = test_data.drop(columns=['label', 'patch_index']).values
    y_test = test_data['label'].values
    test_indices = test_data['patch_index'].values


    X_train = X_train.reshape((-1, dim1, dim2, dim3))
    X_val = X_val.reshape((-1, dim1, dim2, dim3))
    X_test = X_test.reshape((-1, dim1, dim2, dim3))

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    train_label_counts = dict(zip(unique_train, counts_train))
    val_label_counts = dict(zip(unique_val, counts_val))
    test_label_counts = dict(zip(unique_test, counts_test))

    print("训练集标签对应数量:", train_label_counts)
    print("验证集标签对应数量:", val_label_counts)
    print("测试集标签对应数量:", test_label_counts)

    return X_train, X_val, X_test, y_train, y_val, y_test, train_indices, val_indices, test_indices

