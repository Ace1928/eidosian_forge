import builtins
from copy import copy
from typing import Iterable, List, Optional, Tuple
import numpy as np
from ray.data._internal.util import _check_pyarrow_version
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
An example datasource that generates ranges of numbers from [0..n).