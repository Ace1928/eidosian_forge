import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def logloss(self, y_true: FloatsT, y_pred: FloatsT) -> float:
    log_yp = self.xp.log(y_pred + 1e-08)
    loss = y_true * log_yp + (1 - y_true) * self.xp.log(1 - y_pred + 1e-08)
    return -loss