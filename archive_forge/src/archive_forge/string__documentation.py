from __future__ import annotations
from typing import (
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.lib import ensure_string_array
from pandas.compat import pa_version_under10p1
from pandas.compat.numpy import function as nv
from pandas.util._decorators import doc
from pandas.core.dtypes.base import (
from pandas.core.dtypes.common import (
from pandas.core import ops
from pandas.core.array_algos import masked_reductions
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.floating import (
from pandas.core.arrays.integer import (
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
from pandas.core.missing import isna

        Convert myself into a pyarrow Array.
        