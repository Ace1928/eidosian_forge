import os
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import pandas as pd
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from triad.collections.dict import ParamDict
from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw
from triad.utils.io import join, url_to_fs
from triad.utils.pandas_like import PD_UTILS
from fugue.dataframe import LocalBoundedDataFrame, LocalDataFrame, PandasDataFrame
def make_parent_dirs(self) -> None:
    self._fs.makedirs(self._fs._parent(self._fs_path), exist_ok=True)