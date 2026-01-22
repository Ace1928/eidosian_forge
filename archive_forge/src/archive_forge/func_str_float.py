import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def str_float(f: float) -> str:
    return f'{f:.15g}'