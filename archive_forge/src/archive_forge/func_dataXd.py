import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
@property
def dataXd(self) -> ArrayXd:
    if self.data.size:
        reshaped = self.data.reshape(self.data_shape)
    else:
        reshaped = self.data.reshape((self.data.shape[0],) + self.data_shape[1:])
    return cast(ArrayXd, reshaped)