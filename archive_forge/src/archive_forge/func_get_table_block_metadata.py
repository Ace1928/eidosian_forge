import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def get_table_block_metadata(table: Union['pyarrow.Table', 'pandas.DataFrame']) -> 'BlockMetadata':
    from ray.data.block import BlockAccessor, BlockExecStats
    stats = BlockExecStats.builder()
    return BlockAccessor.for_block(table).get_metadata(input_files=None, exec_stats=stats.build())