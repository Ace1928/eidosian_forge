import sys
import types
import numpy as np
import pandas as pd
from .. import util
from ..dimension import Dimension, asdim, dimension_name
from ..element import Element
from ..ndmapping import NdMapping, item_check, sorted_context
from .grid import GridInterface
from .interface import DataError, Interface
from .util import dask_array_module, finite_range

        Given a dataset object and data in the appropriate format for
        the interface, return a simple scalar.
        