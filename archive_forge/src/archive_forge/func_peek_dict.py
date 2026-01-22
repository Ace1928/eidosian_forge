from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
from triad.collections.schema import Schema
from triad.utils.rename import normalize_names
from .._utils.registry import fugue_plugin
from .dataframe import AnyDataFrame, DataFrame, as_fugue_df
@fugue_plugin
def peek_dict(df: AnyDataFrame) -> Dict[str, Any]:
    """Peek the first row of any dataframe as a array

    :param df: the object that can be recognized as a dataframe by Fugue
    :return: the first row as a dict
    """
    return as_fugue_df(df).peek_dict()