import copy
import numbers
from collections.abc import MutableMapping
from warnings import warn
import numpy as np
from nibabel.affines import apply_affine
from .array_sequence import ArraySequence
@data_per_streamline.setter
def data_per_streamline(self, value):
    self._data_per_streamline = LazyDict(value)