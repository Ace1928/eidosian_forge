import io
import pathlib
import posixpath
import warnings
from typing import (
import numpy as np
import ray
from ray._private.utils import _add_creatable_buckets_param_if_s3_uri
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.util import (
from ray.data.block import Block, BlockAccessor
from ray.data.context import DataContext
from ray.data.datasource.block_path_provider import BlockWritePathProvider
from ray.data.datasource.datasource import Datasource, ReadTask, WriteResult
from ray.data.datasource.file_meta_provider import (
from ray.data.datasource.filename_provider import (
from ray.data.datasource.partitioning import (
from ray.data.datasource.path_util import (
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
def read_files(read_paths: Iterable[str]) -> Iterable[Block]:
    nonlocal filesystem, open_stream_args, partitioning
    DataContext._set_current(ctx)
    fs = _unwrap_s3_serialization_workaround(filesystem)
    for read_path in read_paths:
        partitions: Dict[str, str] = {}
        if partitioning is not None:
            parse = PathPartitionParser(partitioning)
            partitions = parse(read_path)
        with _open_file_with_retry(read_path, lambda: open_input_source(fs, read_path, **open_stream_args)) as f:
            for block in read_stream(f, read_path):
                if partitions:
                    block = _add_partitions(block, partitions)
                if self._include_paths:
                    block_accessor = BlockAccessor.for_block(block)
                    block = block_accessor.append_column('path', [read_path] * block_accessor.num_rows())
                yield block