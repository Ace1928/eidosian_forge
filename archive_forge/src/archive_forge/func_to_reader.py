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
def to_reader(self, max_chunksize: Optional[int]=None):
    """
        Convert the Table to a RecordBatchReader.

        Note that this method is zero-copy, it merely exposes the same data under a different API.

        Args:
            max_chunksize (`int`, defaults to `None`)
                Maximum size for RecordBatch chunks. Individual chunks may be smaller depending
                on the chunk layout of individual columns.

        Returns:
            `pyarrow.RecordBatchReader`
        """
    return self.table.to_reader(max_chunksize=max_chunksize)