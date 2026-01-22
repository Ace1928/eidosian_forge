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
def write_from_pcollection(self, pcoll_examples):
    """Add the final steps of the beam pipeline: write to parquet files."""
    import apache_beam as beam

    def inc_num_examples(example):
        beam.metrics.Metrics.counter(self._namespace, 'num_examples').inc()
    _ = pcoll_examples | 'Count N. Examples' >> beam.Map(inc_num_examples)
    return pcoll_examples | 'Get values' >> beam.Values() | 'Save to parquet' >> beam.io.parquetio.WriteToParquet(self._parquet_path, self._schema, shard_name_template='-SSSSS-of-NNNNN.parquet')