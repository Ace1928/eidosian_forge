import numpy as np
from scipy import linalg
from scipy.linalg import toeplitz
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.datasets import sunspots
from statsmodels.tsa.arima_process import (
def invertibleroots(ma):
    proc = ArmaProcess(ma=ma)
    return proc.invertroots(retnew=False)