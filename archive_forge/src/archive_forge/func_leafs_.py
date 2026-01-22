from enum import IntEnum
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
@property
def leafs_(self):
    if self._leafs_ is None:
        raise RuntimeError('NgramPart was not initialized.')
    return self._leafs_