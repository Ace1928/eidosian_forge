from __future__ import annotations
import math
import re
import numpy as np
def svg_1d(chunks, sizes=None, **kwargs):
    return svg_2d(((1,),) + chunks, **kwargs)