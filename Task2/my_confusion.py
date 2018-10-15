import numpy as np

def my_confusion(Ctrues, Cpreds):
    true_labels = Ctrues.astype(int)
    pred_labels = Cpreds.astype(int)
    N = np.size(np.unique(Ctrues))
    CM = np.zeros((N,N))
    for i in range(np.size(true_labels)):
        # The 1 is subtracted because the labels start at 1 rather than 0
        row = true_labels[i][0] - 1
        col = pred_labels[i][0] - 1
        CM[row][col] += 1

    acc = np.trace(CM)/np.size(true_labels)
    return (CM, acc)
