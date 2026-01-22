import base64
import collections
import io
import itertools
import logging
import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING
import fsspec.core
from ..asyn import AsyncFileSystem
from ..callbacks import DEFAULT_CALLBACK
from ..core import filesystem, open, split_protocol
from ..utils import isfilelike, merge_offset_ranges, other_paths
@lru_cache(maxsize=self.cache_size)
def open_refs(field, record):
    """cached parquet file loader"""
    path = self.url.format(field=field, record=record)
    data = io.BytesIO(self.fs.cat_file(path))
    df = self.pd.read_parquet(data, engine='fastparquet')
    refs = {c: df[c].values for c in df.columns}
    return refs