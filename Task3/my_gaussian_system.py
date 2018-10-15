from data_utils import load_data
from my_gaussian_classify import my_gaussian_classify as gaussian
from my_confusion import my_confusion as conf

import numpy as np
import time
import scipy.io as sc


[Xtrn, Ctrn, Xtst, Ctst] = load_data()
start = time.clock()
[Cpreds, Ms, Covs] = gaussian(Xtrn, Ctrn, Xtst, 0.01)
end = time.clock()
print "Time taken: ", (end - start)

[cm, acc] = conf(Ctst, Cpreds)
sc.savemat("cm", {"Confusion_matrix" : cm})
sc.savemat("m26", {"Mean_vector" : Ms[: ,25]})
sc.savemat("cov26", {"Covariance_matrix" : Covs[:, : ,25]})

N = np.size(Ctst)
print "Number of samples: ", N
print "Number of misclassifications: ", int(N - np.trace(cm))
print "Accuracy: ", acc
