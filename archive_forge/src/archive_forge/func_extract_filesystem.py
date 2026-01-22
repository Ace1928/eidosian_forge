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
def extract_filesystem(cls, urlpath, filesystem, dataset_options, open_file_options, storage_options):
    """Extract filesystem object from urlpath or user arguments

        This classmethod should only be overridden for engines that need
        to handle filesystem implementations other than ``fsspec``
        (e.g. ``pyarrow.fs.S3FileSystem``).

        Parameters
        ----------
        urlpath: str or List[str]
            Source directory for data, or path(s) to individual parquet files.
        filesystem: "fsspec" or fsspec.AbstractFileSystem
            Filesystem backend to use. Default is "fsspec"
        dataset_options: dict
            Engine-specific dataset options.
        open_file_options: dict
            Options to be used for file-opening at read time.
        storage_options: dict
            Options to be passed on to the file-system backend.

        Returns
        -------
        fs: Any
            A global filesystem object to be used for metadata
            processing and file-opening by the engine.
        paths: List[str]
            List of data-source paths.
        dataset_options: dict
            Engine-specific dataset options.
        open_file_options: dict
            Options to be used for file-opening at read time.
        """
    if filesystem is None:
        fs = dataset_options.pop('fs', 'fsspec')
    else:
        if 'fs' in dataset_options:
            raise ValueError("Cannot specify a filesystem argument if the 'fs' dataset option is also defined.")
        fs = filesystem
    if fs in (None, 'fsspec'):
        fs, _, paths = get_fs_token_paths(urlpath, mode='rb', storage_options=storage_options)
        return (fs, paths, dataset_options, open_file_options)
    else:
        if not isinstance(fs, AbstractFileSystem):
            raise ValueError(f"Expected fsspec.AbstractFileSystem or 'fsspec'. Got {fs}")
        if storage_options:
            raise ValueError(f'Cannot specify storage_options when an explicit filesystem object is specified. Got: {storage_options}')
        if isinstance(urlpath, (list, tuple, set)):
            if not urlpath:
                raise ValueError('empty urlpath sequence')
            urlpath = [stringify_path(u) for u in urlpath]
        else:
            urlpath = [stringify_path(urlpath)]
        paths = expand_paths_if_needed(urlpath, 'rb', 1, fs, None)
        return (fs, [fs._strip_protocol(u) for u in paths], dataset_options, open_file_options)