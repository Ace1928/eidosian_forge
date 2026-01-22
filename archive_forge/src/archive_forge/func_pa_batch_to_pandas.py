import json
import pickle
from datetime import date, datetime
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import io
import numpy as np
import pandas as pd
import pyarrow as pa
from packaging import version
from pandas.core.dtypes.base import ExtensionDtype
from pyarrow.compute import CastOptions, binary_join_element_wise
from pyarrow.json import read_json, ParseOptions as JsonParseOptions
from triad.constants import TRIAD_VAR_QUOTE
from .convert import as_type
from .iter import EmptyAwareIterable, Slicer
from .json import loads_no_dup
from .schema import move_to_unquoted, quote_name, unquote_name
from .assertion import assert_or_throw
def pa_batch_to_pandas(batch: pa.RecordBatch, use_extension_types: bool=False, use_arrow_dtype: bool=False, **kwargs: Any) -> pd.DataFrame:
    """Convert a pyarrow record batch to pandas dataframe

    :param batch: the pyarrow record batch
    :param use_extension_types: whether to use pandas extension
        data types, default to False
    :param use_arrow_dtype: if True and when pandas supports ``ArrowDType``,
        use pyarrow types, default False
    :param kwargs: other arguments for ``pa.Table.to_pandas``
    :return: the pandas dataframe
    """
    use_arrow_dtype = use_arrow_dtype and hasattr(pd, 'ArrowDtype')
    mapper = partial(to_pandas_types_mapper, use_extension_types=use_extension_types, use_arrow_dtype=use_arrow_dtype)
    return batch.to_pandas(types_mapper=mapper, **kwargs)