import numpy as np
import scipy.io as sc

def load_data():
    dataset = sc.loadmat("/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1661552/data.mat")['dataset'][0][0]
    train = dataset[0][0]
    test = dataset[1][0]

    train_images = np.array(train["images"][0], dtype=float)/255
    train_labels = train["labels"][0]

    test_images = np.array(test["images"][0], dtype=float)/255
    test_labels = test["labels"][0]

    return (train_images, train_labels, test_images, test_labels)
