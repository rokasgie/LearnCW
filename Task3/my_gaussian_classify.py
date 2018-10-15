import numpy as np
import numpy.matlib as matlib

BEST_EPSILON = 0.03

def my_mean(matrix):
    [r, c] = np.shape(matrix)
    if r == 0:
        return np.zeros((1, c))
    else:
        mean_vector = []
        for i in range(c):
            mean_vector.append(np.sum(matrix[:, i])/float(r))

    return np.expand_dims(np.array(mean_vector), axis=0)

def cov_matrix(class_vectors, mean_vector):
    [r, c] = np.shape(class_vectors)
    if r == 0:
        return np.zeros((c, c))
    else:
        M = matlib.repmat(mean_vector, r, 1)
        matrix = class_vectors - M
        return np.dot(matrix.transpose(), matrix)/float(r)

def get_means_covs(training_data, training_labels, N, epsilon):
    means = []
    class_cov_list = []
    for i in range(N):
        indices = np.where(np.in1d(training_labels, i+1))[0]
        class_vectors = np.take(training_data, indices, axis=0)

        mean = my_mean(class_vectors)
        means.append(mean)
        cov = cov_matrix(class_vectors, mean)
        class_cov_list.append(cov + np.identity(np.shape(cov)[0])*epsilon)

    return (np.squeeze(np.array(means), axis=1).transpose(), np.array(class_cov_list).transpose())

def get_dets(covariances):
    invs = []
    dets = []
    for i in range(np.shape(covariances)[2]):
        invs.append(np.linalg.inv(covariances[:, :, i]))
        [_, det] = np.linalg.slogdet(covariances[:, :, i])
        dets.append(det)

    return (np.array(invs).transpose(), np.expand_dims(np.array(dets), axis=0).transpose())


def my_gaussian(Xtst, mean, inv, log_det, prior):
    sub = Xtst - mean
    mul = np.dot(sub, inv)
    matrix = []

    for i in range(np.shape(mul)[0]):
        matrix.append(np.dot(mul[i, :], Xtst[i, :].transpose()))

    return -0.5*(np.array(matrix) + log_det) + np.log(prior)


def my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon=None):
    if epsilon == None:
        epsilon = BEST_EPSILON

    [labels, counts] = np.unique(Xtrn, return_counts=True)
    N = np.size(labels)
    total = np.sum(np.shape(Ctrn)[0])
    [Ms, Covs] = get_means_covs(Xtrn, Ctrn, N, epsilon)
    [invs, log_dets] = get_dets(Covs)

    prob_matrix = []
    for i in range(N):
        prob_matrix.append(my_gaussian(Xtst, Ms[:, i], invs[:, :, i], log_dets[i], counts[i]/float(total)))

    Cpreds = np.expand_dims(np.argmax(np.array(prob_matrix), axis=0) + 1, axis=1)
    return (Cpreds, Ms, Covs)
