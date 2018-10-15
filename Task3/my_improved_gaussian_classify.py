from my_gaussian_classify import my_gaussian_classify
from my_gaussian_classify import my_mean
from my_confusion import my_confusion as conf

import numpy as np
import numpy.matlib as matlib


NUMBER_OF_KMEANS_REPS = 3
BEST_NUMBER_OF_CLUSTERS = 8
NUM_OF_CLUSTERS_TO_TEST = [1, 3, 6, 8]

def update_labels(Ctrn, new_labels, class_no, N):
    new_Ctrn = []
    j = 0
    for i in Ctrn:
        if i[0] == class_no:
            if new_labels[j] != 0:
                new_Ctrn.append(new_labels[j] + N)
            else:
                new_Ctrn.append(i[0])
            j += 1
        else:
            new_Ctrn.append(i[0])

    return np.expand_dims(np.array(new_Ctrn), axis = 1)

def normalize_preds(preds, class_no, N):
    for i in range(np.size(preds)):
        if preds[i] > N:
            preds[i] = class_no
    return preds

def experiment(Xtrn, Ctrn, Xtst, Ctst):
    classes = np.unique(Ctrn)
    N = np.size(classes)
    Kns_vector = []
    for i in classes:
        acc_list = []
        labels_list = []
        for j in NUM_OF_CLUSTERS_TO_TEST:
            labels = my_kmeans_classify(get_class_vectors(Xtrn, Ctrn, i), j)
            new_Ctrn = update_labels(Ctrn, labels, i, N)
            [Cpreds, _, _] = my_gaussian_classify(Xtrn, new_Ctrn, Xtst)
            preds = normalize_preds(Cpreds, i, N)
            [_, acc] = conf(Ctst, preds)
            acc_list.append(acc)

        max = np.argmax(acc_list)
        Kns_vector.append(NUM_OF_CLUSTERS_TO_TEST[max])

    return np.array(Kns_vector).transpose()

def mul_by_transpose(X, size):
    result = []
    for i in range(size):
        result.append(np.dot(X[i], np.transpose(X[i])))
    return np.expand_dims(np.array(result).transpose(), axis=1)

def dists(Xtrn, Xtst):
    N = np.shape(Xtst)[0]
    M = np.shape(Xtrn)[0]

    XX = mul_by_transpose(Xtst, N)
    repeatedXX =  matlib.repmat(XX, 1, M)

    YY = mul_by_transpose(Xtrn, M)
    repeatedYY = matlib.repmat(YY, 1, N)

    mul = np.dot(Xtst, Xtrn.transpose())
    return np.add(np.subtract(repeatedXX, 2*mul), np.transpose(repeatedYY))

def get_class_vectors(Xtrn, Ctrn, class_no):
    indices = np.where(np.in1d(Ctrn, class_no))[0]
    return np.take(Xtrn, indices, axis=0)

def init_centres(K, dims):
    rand_vec = []
    for i in range(K):
        rand_vec.append(np.random.rand(1, dims))
    return np.squeeze(np.array(rand_vec), axis=1)

def mean_squared_error(Xtrn, Ks, idx):
    sum = 0
    for i in range(np.shape(Ks)[0]):
        cluster_vectors = get_class_vectors(Xtrn, idx, i)
        sum += np.sum(np.subtract(cluster_vectors, my_mean(cluster_vectors))**2)
    return sum/float(np.shape(Xtrn)[0])

def kmeans(Xtrn, num_of_Ks):
    Ks = init_centres(num_of_Ks, np.shape(Xtrn)[1])
    Ks_prev = np.zeros(np.shape(Ks))
    while not(np.array_equal(Ks, Ks_prev)):
        Ks_prev = Ks
        dist = dists(Xtrn, Ks)
        idx = np.argmin(dist, axis=0)
        centres = []
        for i in range(num_of_Ks):
            class_vectors = get_class_vectors(Xtrn, idx, i)
            if np.shape(class_vectors)[0] != 0:
                centres.append(my_mean(class_vectors))
        Ks = np.squeeze(np.array(centres), axis=1)

    return (Ks, idx)

def my_kmeans_classify(Xtrn, num_of_Ks):
    idx_list = []
    errors = []
    for j in range(NUMBER_OF_KMEANS_REPS):
        [Ks, idx] = kmeans(Xtrn, num_of_Ks)
        error = mean_squared_error(Xtrn, Ks, idx)
        errors.append(error)
        idx_list.append(idx)

    return idx_list[np.argmin(errors)]

def my_improved_gaussian_classify(Xtrn, Ctrn, Xtst, Kns=None):
    N = np.size(np.unique(Ctrn))
    if Kns == None:
        Kns = np.empty((N)).astype(int)
        Kns.fill(BEST_NUMBER_OF_CLUSTERS)

    vec_list = []
    labels_list = []
    real_labels_dict = {}
    l = 1
    for i in range(N):
        class_vectors = get_class_vectors(Xtrn, Ctrn, i+1)
        labels = my_kmeans_classify(class_vectors, Kns[i]) + l
        l = np.max(labels) + 1
        vec_list.append(class_vectors)
        labels_list.append(labels)

        for j in labels:
            real_labels_dict[j] = i+1

    new_Xtrn = np.reshape(np.array(vec_list), np.shape(Xtrn))
    new_Ctrn = np.reshape(np.array(labels_list), (np.shape(Ctrn)))
    [Cpreds, Ms, Covs] = my_gaussian_classify(new_Xtrn, new_Ctrn, Xtst)

    real_preds = []
    for i in Cpreds:
        real_preds.append(real_labels_dict[int(i)])

    preds = np.expand_dims(np.array(real_preds), axis=1)
    return preds
