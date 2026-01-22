from functools import lru_cache
from typing import Dict, Generic, Tuple, Type, TypeVar, Union, cast
import numpy as np
from scipy.sparse import (
from pennylane.data.base.attribute import AttributeInfo, DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group
Returns a dict mapping sparse array class names to the class.