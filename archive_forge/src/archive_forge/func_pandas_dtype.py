from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.collections.dict import IndexedOrderedDict
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.pandas_like import PD_UTILS
from triad.utils.pyarrow import (
from triad.utils.schema import (
@property
def pandas_dtype(self) -> Dict[str, np.dtype]:
    """Convert as `dtype` dict for pandas dataframes.
        Currently, struct type is not supported
        """
    return self.to_pandas_dtype(self.pa_schema)