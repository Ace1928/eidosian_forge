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
@property
def scalar_type(self):
    """Returns the type of the underlying tensor elements."""
    data_field_index = self.storage_type.get_field_index('data')
    return self.storage_type[data_field_index].type.value_type