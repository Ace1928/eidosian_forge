import numpy as np
from scipy.optimize import minimize
import GPy
from GPy.kern import Kern
from GPy.core import Param
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
def select_length(Xraw, yraw, bounds, num_f):
    """Select the number of datapoints to keep, using cross validation"""
    min_len = 200
    if Xraw.shape[0] < min_len:
        return Xraw.shape[0]
    else:
        length = min_len - 10
        scores = []
        while length + 10 <= Xraw.shape[0]:
            length += 10
            base_vals = np.array(list(bounds.values())).T
            X_len = Xraw[-length:, :]
            y_len = yraw[-length:]
            oldpoints = X_len[:, :num_f]
            old_lims = np.concatenate((np.max(oldpoints, axis=0), np.min(oldpoints, axis=0))).reshape(2, oldpoints.shape[1])
            limits = np.concatenate((old_lims, base_vals), axis=1)
            X = normalize(X_len, limits)
            y = standardize(y_len).reshape(y_len.size, 1)
            kernel = TV_SquaredExp(input_dim=X.shape[1], variance=1.0, lengthscale=1.0, epsilon=0.1)
            m = GPy.models.GPRegression(X, y, kernel)
            m.optimize(messages=True)
            scores.append(m.log_likelihood())
        idx = np.argmax(scores)
        length = (idx + int(min_len / 10)) * 10
        return length