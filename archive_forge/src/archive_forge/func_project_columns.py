from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Literal
from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path
from dask.base import compute_as_if_collection, tokenize
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.core import DataFrame, Scalar
from dask.dataframe.io.io import from_map
from dask.dataframe.io.orc.utils import ORCEngine
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.highlevelgraph import HighLevelGraph
from dask.utils import apply
def project_columns(self, columns):
    """Return a new ORCFunctionWrapper object with
        a sub-column projection.
        """
    if columns == self.columns:
        return self
    func = copy.deepcopy(self)
    func._columns = columns
    return func