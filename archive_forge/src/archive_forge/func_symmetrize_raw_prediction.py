from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit
from scipy.stats import gmean
from ..utils.extmath import softmax
def symmetrize_raw_prediction(self, raw_prediction):
    return raw_prediction - np.mean(raw_prediction, axis=1)[:, np.newaxis]