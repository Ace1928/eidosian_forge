import itertools
import math
from typing import Sequence, Tuple, Union
import numpy as np
from onnx.reference.op_run import OpRun
def lp_pool_p(x):
    return lp_pool(x, p)