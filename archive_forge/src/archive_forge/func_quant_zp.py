from itertools import product
import math
import random
import time
import einops
import numpy as np
import pytest
from scipy.stats import norm
import torch
import bitsandbytes as bnb
from bitsandbytes import functional as F
from tests.helpers import (
def quant_zp(x):
    dtype = x.dtype
    x = x.float()
    dyna = x.max() - x.min()
    if dyna == 0:
        dyna = 1
    qx = 254.0 / dyna
    minx = x.min()
    zpx = torch.round(x.min() * qx) - 127
    x = qx * x + zpx
    return (x, qx, zpx)