from data_utils import load_data
from my_improved_gaussian_classify import my_improved_gaussian_classify as improved_gaussian
from my_improved_gaussian_classify import experiment
from my_confusion import my_confusion as conf

import time
import numpy as np
import scipy.io as sc

NUMBER_OF_KMEANS_REPS = 3
BEST_EXPERIMENT_KNS = [1, 2, 1, 6, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 6, 2, 1, 2, 1, 2, 1]


[Xtrn, Ctrn, Xtst, Ctst] = load_data()
start = time.clock()
Kns = experiment(Xtrn, Ctrn, Xtst, Ctst)
print Kns
print "Experiment time: ", time.clock() - start

start = time.clock()
Cpreds = improved_gaussian(Xtrn, Ctrn, Xtst, Kns)
end = time.clock()
print "Time taken: ", (end - start)

[cm, acc] = conf(Ctst, Cpreds)
sc.savemat("cm_improved", {"Confusion_matrix" : cm})

N = np.size(Ctst)
print "Number of samples: ", N
print "Number of misclassifications: ", int(N - np.trace(cm))
print "Accuracy: ", acc
