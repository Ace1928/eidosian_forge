import math
import numpy as np
from tensorflow.python.ops.distributions import special_math
def normal_pdf(x):
    return math.exp(-x ** 2 / 2.0) / math.sqrt(2 * math.pi)