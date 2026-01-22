from __future__ import annotations
import math
import re
import numpy as np
def ratio_response(x):
    """How we display actual size ratios

    Common ratios in sizes span several orders of magnitude,
    which is hard for us to perceive.

    We keep ratios in the 1-3 range accurate, and then apply a logarithm to
    values up until about 100 or so, at which point we stop scaling.
    """
    if x < math.e:
        return x
    elif x <= 100:
        return math.log(x + 12.4)
    else:
        return math.log(100 + 12.4)