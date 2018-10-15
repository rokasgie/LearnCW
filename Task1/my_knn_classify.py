import numpy as np
import numpy.matlib as matlib
from scipy.stats import mode


def mul_by_transpose(X, size):
    result = []
    for i in range(size):
        result.append(np.dot(X[i], np.transpose(X[i])))
    return np.expand_dims(np.array(result).transpose(), axis=1)

def dists(Xtrn, Xtst):
    [M, _] = np.shape(Xtrn)
    [N, _] = np.shape(Xtst)
    mul = np.dot(Xtst, Xtrn.transpose())

    XX = mul_by_transpose(Xtst, N)
    YY = mul_by_transpose(Xtrn, M)

    return np.add(np.subtract(matlib.repmat(XX, 1, M), 2*mul), (matlib.repmat(YY, 1, N)).transpose())

def my_knn_classify(Xtrn, Ctrn, Xtst, Ks):
    distance_matrix = dists(Xtrn, Xtst)
    closest_points = np.argsort(distance_matrix)[:, :np.max(Ks)]
    closest_labels = np.take(Ctrn, closest_points)

    labels = []
    for i in range(np.size(Ks)):
        labels.append(mode(closest_labels[:, :Ks[i]], axis=1)[0])

    return np.squeeze(np.array(labels), axis=2).transpose()
