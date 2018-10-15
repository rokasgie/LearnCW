import numpy as np
BEST_EPSILON_VALUE = 0.2


def calc_sum(bXtrn, Ctrn):
    N = np.size(np.unique(Ctrn))
    [rows, cols] = np.shape(bXtrn)
    sum_matrix = np.zeros((cols, N))
    for i in range(cols):
        v = bXtrn[:, i]
        for j in range(N):
            indices = np.where(np.in1d(Ctrn, j+1))[0]
            sum_matrix[i][j] = np.sum(np.take(v, indices))

    return sum_matrix

def prep_data(Xtrn, Xtst, threshold):
    bXtrn = (Xtrn > threshold).astype(int)
    bXtst = (Xtst > threshold).astype(int)
    return (bXtrn, bXtst)

def get_freq_matrix(sum_matrix, Ctrn):
    [labels, counts] = np.unique(Ctrn, return_counts=True)
    freq_matrix = []
    for i in range(np.shape(sum_matrix)[1]):
        freq_matrix.append(sum_matrix[:, i]/float(counts[i]))

    return np.array(freq_matrix).transpose()

def get_log(v):
    if v == 0:
        return np.log(1.0E-10)
    else:
        return np.log(v)

def vectorize_freq(freq_matrix):
    [rows, cols] = np.shape(freq_matrix)
    vectorized = np.empty((rows*2, cols))
    for j in range(cols):
        for i in range(rows):
            vectorized[2*i][j] = get_log(1 - freq_matrix[i][j])
            vectorized[2*i + 1][j] = get_log(freq_matrix[i][j])

    return vectorized

def vectorize_test(bXtst):
    [_, cols] = np.shape(bXtst)
    result = []
    for i in range(cols):
        result.append(1 - bXtst[:, i])
        result.append(bXtst[:, i])

    return np.array(result).transpose()

def my_bnb_classify(Xtrn, Ctrn, Xtst, threshold=None):
    if threshold == None:
        threshold = BEST_EPSILON_VALUE

    [bXtrn, bXtst] = prep_data(Xtrn, Xtst, threshold)
    sum_matrix = calc_sum(bXtrn, Ctrn)
    freq_matrix = get_freq_matrix(sum_matrix, Ctrn)

    test = vectorize_test(bXtst)
    freq = vectorize_freq(freq_matrix)

    labels = np.argmax(np.dot(test, freq), axis=1)
    return np.expand_dims(labels, axis=1) + 1
