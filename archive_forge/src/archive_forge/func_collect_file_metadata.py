from __future__ import annotations
import re
import warnings
import numpy as np
import pandas as pd
from fsspec.core import expand_paths_if_needed, get_fs_token_paths, stringify_path
from fsspec.spec import AbstractFileSystem
from dask import config
from dask.dataframe.io.utils import _is_local_fs
from dask.utils import natural_sort_key, parse_bytes
@classmethod
def collect_file_metadata(cls, path, fs, file_path):
    """
        Collect parquet metadata from a file and set the file_path.

        Parameters
        ----------
        path: str
            Parquet-file path to extract metadata from.
        fs: FileSystem
        file_path: str
            Relative path to set as `file_path` in the metadata.

        Returns
        -------
        A metadata object.  The specific type should be recognized
        by the aggregate_metadata method.
        """
    raise NotImplementedError()