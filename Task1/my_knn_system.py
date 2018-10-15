from data_utils import load_data
from my_knn_classify import my_knn_classify as knn
from my_confusion import my_confusion as conf

import time
import numpy as np
import scipy.io as sc

Ks = [1, 3, 5, 10, 20]


[Xtrn, Ctrn, Xtst, Ctst] = load_data()
start = time.clock()
Cpreds = knn(Xtrn, Ctrn, Xtst, Ks)
end = time.clock()
print "Time taken: ", (end - start)

N = np.size(Ctst)
for i in range(np.size(Ks)):
    [CM, acc] = conf(Ctst, Cpreds[:, i].reshape(N, 1))

    print "Number of nearest neighbours: ", Ks[i]
    print "Number of samples: ", N
    print "Number of misclassifications: ", int(N - np.trace(CM))
    print "Accuracy: ", acc
    print "--------------------------------"

    filename = "cm" +  str(Ks[i])
    sc.savemat(filename, {"Confusion_matrix" : CM})
