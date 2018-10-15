from data_utils import load_data
from my_bnb_classify import my_bnb_classify as bnb
from my_confusion import my_confusion as conf

import numpy as np
import time
import scipy.io as sc


[Xtrn, Ctrn, Xtst, Ctst] = load_data()
start = time.clock()
Cpreds =  bnb(Xtrn, Ctrn, Xtst, 1)
print "Time taken: ", (time.clock() - start)

[cm, acc] = conf(Ctst, Cpreds)
sc.savemat("cm", {"Confusion_matrix" : cm})

N = np.size(Ctst)
print "Number of samples: ", N
print "Number of misclassifications: ", int(N - np.trace(cm))
print "Accuracy: ", acc
