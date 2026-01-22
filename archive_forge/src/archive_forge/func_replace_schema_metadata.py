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
def replace_schema_metadata(self, *args, **kwargs):
    """
        EXPERIMENTAL: Create shallow copy of table by replacing schema
        key-value metadata with the indicated new metadata (which may be `None`,
        which deletes any existing metadata).

        Args:
            metadata (`dict`, defaults to `None`):

        Returns:
            `datasets.table.Table`: shallow_copy
        """
    table = self.table.replace_schema_metadata(*args, **kwargs)
    blocks = []
    for tables in self.blocks:
        blocks.append([t.replace_schema_metadata(*args, **kwargs) for t in tables])
    return ConcatenationTable(table, self.blocks)