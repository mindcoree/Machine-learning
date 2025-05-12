import numpy as np

# root mean square error

def rmse(labels,predictions):
    n = len(labels)
    difference = np.subtract(labels,predictions)
    return np.sqrt(np.dot(difference,difference) / n)


