import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def to_pickable(x):
    sign, man, exp, bc = x
    return (sign, hex(man)[2:], exp, bc)