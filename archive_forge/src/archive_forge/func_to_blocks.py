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
def to_blocks(table: Union[pa.Table, Table]) -> List[List[TableBlock]]:
    if isinstance(table, pa.Table):
        return [[InMemoryTable(table)]]
    elif isinstance(table, ConcatenationTable):
        return copy.deepcopy(table.blocks)
    else:
        return [[table]]