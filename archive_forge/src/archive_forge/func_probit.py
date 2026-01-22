import math
import numpy as np
from tensorflow.python.ops.distributions import special_math
def probit(x):
    return special_math.ndtri(x)