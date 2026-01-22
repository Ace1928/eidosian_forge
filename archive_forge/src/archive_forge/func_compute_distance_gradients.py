import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter
import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors
def compute_distance_gradients(self):
    """Compute and store partial derivatives of poincare distance d(u, v) w.r.t all u and all v."""
    if self._distance_gradients_computed:
        return
    self.compute_distances()
    euclidean_dists_squared = self.euclidean_dists ** 2
    c_ = (4 / (self.alpha * self.beta * np.sqrt(self.gamma ** 2 - 1)))[:, np.newaxis, :]
    u_coeffs = ((euclidean_dists_squared + self.alpha) / self.alpha)[:, np.newaxis, :]
    distance_gradients_u = u_coeffs * self.vectors_u - self.vectors_v
    distance_gradients_u *= c_
    nan_gradients = self.gamma == 1
    if nan_gradients.any():
        distance_gradients_u.swapaxes(1, 2)[nan_gradients] = 0
    self.distance_gradients_u = distance_gradients_u
    v_coeffs = ((euclidean_dists_squared + self.beta) / self.beta)[:, np.newaxis, :]
    distance_gradients_v = v_coeffs * self.vectors_v - self.vectors_u
    distance_gradients_v *= c_
    if nan_gradients.any():
        distance_gradients_v.swapaxes(1, 2)[nan_gradients] = 0
    self.distance_gradients_v = distance_gradients_v
    self._distance_gradients_computed = True