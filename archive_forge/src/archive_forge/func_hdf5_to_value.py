import json
import typing
from functools import lru_cache
from typing import Dict, FrozenSet, Generic, List, Type, TypeVar
import numpy as np
import pennylane as qml
from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group, h5py
from pennylane.operation import Operator, Tensor
from ._wires import wires_to_json
def hdf5_to_value(self, bind: HDF5Group) -> Op:
    return self._hdf5_to_ops(bind)[0]