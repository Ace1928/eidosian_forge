import logging
import os
from typing import Any, Callable, Dict, List, Optional, Type, Union
import numpy as np
import pandas as pd
from triad import Schema
from triad.collections.dict import IndexedOrderedDict
from triad.utils.assertion import assert_or_throw
from triad.utils.io import makedirs
from triad.utils.pandas_like import PandasUtils
from fugue._utils.io import load_df, save_df
from fugue._utils.misc import import_fsql_dependency
from fugue.collections.partition import (
from fugue.collections.sql import StructuredRawSQL
from fugue.constants import KEYWORD_PARALLELISM, KEYWORD_ROWCOUNT
from fugue.dataframe import (
from fugue.dataframe.dataframe import as_fugue_df
from fugue.dataframe.utils import get_join_schemas
from .execution_engine import (
@property
def pl_utils(self) -> PandasUtils:
    """Pandas-like dataframe utils"""
    return PandasUtils()