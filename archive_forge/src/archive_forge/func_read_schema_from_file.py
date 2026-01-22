import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
def read_schema_from_file(filename: str) -> pa.Schema:
    """
    Infer arrow table schema from file without loading whole file into memory.
    Usefull especially while having very big files.
    """
    with pa.memory_map(filename) as memory_mapped_stream:
        schema = pa.ipc.open_stream(memory_mapped_stream).schema
    return schema