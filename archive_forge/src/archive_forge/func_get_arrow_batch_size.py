import os
import posixpath
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union
import numpy as np
import pyarrow as pa
import datasets
from datasets.arrow_writer import ArrowWriter, ParquetWriter
from datasets.config import MAX_SHARD_SIZE
from datasets.filesystems import (
from datasets.iterable_dataset import _BaseExamplesIterable
from datasets.utils.py_utils import convert_file_size_to_int
def get_arrow_batch_size(it):
    for batch in it:
        yield pa.RecordBatch.from_pydict({'batch_bytes': [batch.nbytes]})