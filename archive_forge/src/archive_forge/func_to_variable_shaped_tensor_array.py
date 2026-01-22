import itertools
import json
import sys
from typing import Iterable, Optional, Tuple, List, Sequence, Union
from pkg_resources._vendor.packaging.version import parse as parse_version
import numpy as np
import pyarrow as pa
from ray.air.util.tensor_extensions.utils import (
from ray._private.utils import _get_pyarrow_version
from ray.util.annotations import PublicAPI
def to_variable_shaped_tensor_array(self) -> 'ArrowVariableShapedTensorArray':
    """
        Convert this tensor array to a variable-shaped tensor array.

        This is primarily used when concatenating multiple chunked tensor arrays where
        at least one chunked array is already variable-shaped and/or the shapes of the
        chunked arrays differ, in which case the resulting concatenated tensor array
        will need to be in the variable-shaped representation.
        """
    return ArrowVariableShapedTensorArray.from_numpy(self.to_numpy())