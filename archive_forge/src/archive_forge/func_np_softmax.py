import os, sys
import numpy as np
import torch
from torch.nn import functional as F
def np_softmax(self, x: np.ndarray, axis: int):
    x -= x.max(axis=axis, keepdims=True)
    e: np.ndarray = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)