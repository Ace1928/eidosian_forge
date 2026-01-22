import numpy as np
from scipy.optimize import minimize
import GPy
from GPy.kern import Kern
from GPy.core import Param
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
def update_gradients_full(self, dL_dK, X, X2):
    if X2 is None:
        X2 = np.copy(X)
    T1 = X[:, 0].reshape(-1, 1)
    T2 = X2[:, 0].reshape(-1, 1)
    X = X[:, 1:]
    X2 = X2[:, 1:]
    dist2 = np.square(euclidean_distances(X, X2)) / self.lengthscale
    dvar = np.exp(-np.square(euclidean_distances(X, X2) / self.lengthscale))
    dl = -(2 * euclidean_distances(X, X2) ** 2 * self.variance * np.exp(-dist2)) * self.lengthscale ** (-2)
    n = pairwise_distances(T1, T2, 'cityblock') / 2
    deps = -n * (1 - self.epsilon) ** (n - 1)
    self.variance.gradient = np.sum(dvar * dL_dK)
    self.lengthscale.gradient = np.sum(dl * dL_dK)
    self.epsilon.gradient = np.sum(deps * dL_dK)