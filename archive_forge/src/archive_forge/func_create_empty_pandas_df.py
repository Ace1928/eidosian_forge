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
def create_empty_pandas_df(self, use_extension_types: bool=False, use_arrow_dtype: bool=False) -> pd.DataFrame:
    """Create an empty pandas dataframe based on the schema

        :param use_extension_types: if True, use pandas extension types,
            default False
        :param use_arrow_dtype: if True and when pandas supports ``ArrowDType``,
            use pyarrow types, default False
        :return: empty pandas dataframe
        """
    dtypes = self.to_pandas_dtype(use_extension_types=use_extension_types, use_arrow_dtype=use_arrow_dtype)
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in dtypes.items()})