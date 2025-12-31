import numpy as np

def calculate_weights(HSI, p, neighborhood_size):
    C, P, _ = HSI.shape
    p2 = p // 2
    center_pixel = HSI[:, p2, p2]

    weight_size = 2 * neighborhood_size + 1
    weights = np.zeros((weight_size, weight_size))

    d_sum = 0
    distance_matrix = np.zeros((weight_size, weight_size))

    for i in range(-neighborhood_size, neighborhood_size + 1):
        for j in range(-neighborhood_size, neighborhood_size + 1):
            if i == 0 and j == 0:
                continue
            neighbor_pixel = HSI[:, p2 + i, p2 + j]
            distance = np.linalg.norm(neighbor_pixel - center_pixel)

            distance_matrix[i + neighborhood_size, j + neighborhood_size] = distance
            d_sum += distance

    if d_sum >= 115694:
        for i in range(-neighborhood_size, neighborhood_size + 1):
            for j in range(-neighborhood_size, neighborhood_size + 1):
                if i == 0 and j == 0:
                    continue

                dist = distance_matrix[i + neighborhood_size, j + neighborhood_size]
                weights[i + neighborhood_size, j + neighborhood_size] = 1.0 / (dist + 1e-6)
    else:
        weights = np.random.uniform(-0.3, 0.3, (weight_size, weight_size))
        weights[neighborhood_size, neighborhood_size] = 0

    weight_sum_val = np.sum(weights)
    if weight_sum_val != 0:
        weights = weights / weight_sum_val

    return weights


def generate_mixed_sample(HSI, p, neighborhood_size=1):
    C, P, _ = HSI.shape
    p2 = p // 2

    original_center_pixel_value = HSI[:, p2, p2]
    offset_value = np.zeros(C)

    weights = calculate_weights(HSI, p, neighborhood_size)

    for i in range(-neighborhood_size, neighborhood_size + 1):
        for j in range(-neighborhood_size, neighborhood_size + 1):
            if i == 0 and j == 0:
                continue

            diff = HSI[:, p2 + i, p2 + j] - original_center_pixel_value
            w = weights[i + neighborhood_size, j + neighborhood_size]
            offset_value += w * diff

    return original_center_pixel_value + offset_value


def augment_hyperspectral_data1(X, y, neighborhood_size=2):
    N, C, P, _ = X.shape
    augmented_data = np.copy(X)

    y = y.astype(int)
    counts = np.bincount(y)
    for label, count in enumerate(counts):
        print(f'类别 {label}: {count} 个')

    for n in range(N):
        new_center_pixel_value = generate_mixed_sample(X[n], P, neighborhood_size)
        augmented_data[n, :, P // 2, P // 2] = new_center_pixel_value

    metrics = {}
    return augmented_data, y