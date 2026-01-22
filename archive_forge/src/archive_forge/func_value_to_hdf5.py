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
def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: Op) -> HDF5Group:
    return self._ops_to_hdf5(bind_parent, key, [value])