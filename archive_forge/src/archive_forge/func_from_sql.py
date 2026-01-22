import contextlib
import copy
import fnmatch
import itertools
import json
import math
import os
import posixpath
import re
import shutil
import sys
import tempfile
import time
import warnings
import weakref
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from functools import partial, wraps
from io import BytesIO
from math import ceil, floor
from pathlib import Path
from random import sample
from typing import (
from typing import Sequence as Sequence_
import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from huggingface_hub import CommitInfo, CommitOperationAdd, CommitOperationDelete, DatasetCard, DatasetCardData, HfApi
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map
from . import config
from .arrow_reader import ArrowReader
from .arrow_writer import ArrowWriter, OptimizedTypedSequence
from .data_files import sanitize_patterns
from .download.streaming_download_manager import xgetsize
from .features import Audio, ClassLabel, Features, Image, Sequence, Value
from .features.features import (
from .filesystems import is_remote_filesystem
from .fingerprint import (
from .formatting import format_table, get_format_type_from_alias, get_formatter, query_table
from .formatting.formatting import LazyDict, _is_range_contiguous
from .info import DatasetInfo, DatasetInfosDict
from .naming import _split_re
from .search import IndexableMixin
from .splits import NamedSplit, Split, SplitDict, SplitInfo
from .table import (
from .tasks import TaskTemplate
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.deprecation_utils import deprecated
from .utils.file_utils import estimate_dataset_size
from .utils.hub import list_files_info, preupload_lfs_files
from .utils.info_utils import is_small_dataset
from .utils.metadata import MetadataConfigs
from .utils.py_utils import (
from .utils.stratify import stratified_shuffle_split_generate_indices
from .utils.tf_utils import dataset_to_tf, minimal_tf_collate_fn, multiprocess_dataset_to_tf
from .utils.typing import ListLike, PathLike
@staticmethod
def from_sql(sql: Union[str, 'sqlalchemy.sql.Selectable'], con: Union[str, 'sqlalchemy.engine.Connection', 'sqlalchemy.engine.Engine', 'sqlite3.Connection'], features: Optional[Features]=None, cache_dir: str=None, keep_in_memory: bool=False, **kwargs):
    """Create Dataset from SQL query or database table.

        Args:
            sql (`str` or `sqlalchemy.sql.Selectable`):
                SQL query to be executed or a table name.
            con (`str` or `sqlite3.Connection` or `sqlalchemy.engine.Connection` or `sqlalchemy.engine.Connection`):
                A [URI string](https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls) used to instantiate a database connection or a SQLite3/SQLAlchemy connection object.
            features ([`Features`], *optional*):
                Dataset features.
            cache_dir (`str`, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`SqlConfig`].

        Returns:
            [`Dataset`]

        Example:

        ```py
        >>> # Fetch a database table
        >>> ds = Dataset.from_sql("test_data", "postgres:///db_name")
        >>> # Execute a SQL query on the table
        >>> ds = Dataset.from_sql("SELECT sentence FROM test_data", "postgres:///db_name")
        >>> # Use a Selectable object to specify the query
        >>> from sqlalchemy import select, text
        >>> stmt = select([text("sentence")]).select_from(text("test_data"))
        >>> ds = Dataset.from_sql(stmt, "postgres:///db_name")
        ```

        <Tip>

        The returned dataset can only be cached if `con` is specified as URI string.

        </Tip>
        """
    from .io.sql import SqlDatasetReader
    return SqlDatasetReader(sql, con, features=features, cache_dir=cache_dir, keep_in_memory=keep_in_memory, **kwargs).read()