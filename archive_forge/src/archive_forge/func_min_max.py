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
def min_max(x):
    maxA = torch.amax(x, dim=2, keepdim=True)
    minA = torch.amin(x, dim=2, keepdim=True)
    scale = (maxA - minA) / 2.0
    return ((127 * (x - minA - scale) / scale).to(torch.int8), minA, scale)