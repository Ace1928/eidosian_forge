from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def random_complexes(N):
    random.seed(1)
    a = []
    for i in range(N):
        x1 = random.uniform(-10, 10)
        y1 = random.uniform(-10, 10)
        x2 = random.uniform(-10, 10)
        y2 = random.uniform(-10, 10)
        z1 = complex(x1, y1)
        z2 = complex(x2, y2)
        a.append((z1, z2))
    return a