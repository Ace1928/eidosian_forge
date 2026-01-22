import errno
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from . import config
from .features import Features, Image, Value
from .features.features import (
from .filesystems import is_remote_filesystem
from .info import DatasetInfo
from .keyhash import DuplicatedKeysError, KeyHasher
from .table import array_cast, cast_array_to_feature, embed_table_storage, table_cast
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import hash_url_to_filename
from .utils.py_utils import asdict, first_non_null_value
def parquet_to_arrow(source, destination) -> List[int]:
    """Convert parquet file to arrow file. Inputs can be str paths or file-like objects"""
    stream = None if isinstance(destination, str) else destination
    parquet_file = pa.parquet.ParquetFile(source)
    with ArrowWriter(schema=parquet_file.schema_arrow, path=destination, stream=stream) as writer:
        for record_batch in parquet_file.iter_batches():
            pa_table = pa.Table.from_batches([record_batch])
            writer.write_table(pa_table)
        num_bytes, num_examples = writer.finalize()
    return (num_bytes, num_examples)