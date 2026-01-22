import contextlib
import copy
import fnmatch
import json
import math
import posixpath
import re
import warnings
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import fsspec
import numpy as np
from huggingface_hub import (
from . import config
from .arrow_dataset import PUSH_TO_HUB_WITHOUT_METADATA_CONFIGS_SPLIT_PATTERN_SHARDED, Dataset
from .features import Features
from .features.features import FeatureType
from .info import DatasetInfo, DatasetInfosDict
from .naming import _split_re
from .splits import NamedSplit, Split, SplitDict, SplitInfo
from .table import Table
from .tasks import TaskTemplate
from .utils import logging
from .utils.deprecation_utils import deprecated
from .utils.doc_utils import is_documented_by
from .utils.hub import list_files_info
from .utils.metadata import MetadataConfigs
from .utils.py_utils import asdict, glob_pattern_to_regex, string_to_dict
from .utils.typing import PathLike
@staticmethod
def from_parquet(path_or_paths: Dict[str, PathLike], features: Optional[Features]=None, cache_dir: str=None, keep_in_memory: bool=False, columns: Optional[List[str]]=None, **kwargs) -> 'DatasetDict':
    """Create [`DatasetDict`] from Parquet file(s).

        Args:
            path_or_paths (`dict` of path-like):
                Path(s) of the CSV file(s).
            features ([`Features`], *optional*):
                Dataset features.
            cache_dir (`str`, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            columns (`List[str]`, *optional*):
                If not `None`, only these columns will be read from the file.
                A column name may be a prefix of a nested field, e.g. 'a' will select
                'a.b', 'a.c', and 'a.d.e'.
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`ParquetConfig`].

        Returns:
            [`DatasetDict`]

        Example:

        ```py
        >>> from datasets import DatasetDict
        >>> ds = DatasetDict.from_parquet({'train': 'path/to/dataset/parquet'})
        ```
        """
    from .io.parquet import ParquetDatasetReader
    return ParquetDatasetReader(path_or_paths, features=features, cache_dir=cache_dir, keep_in_memory=keep_in_memory, columns=columns, **kwargs).read()