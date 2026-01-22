import numpy as np
import pytest
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
Check error when n_components <= n_samples