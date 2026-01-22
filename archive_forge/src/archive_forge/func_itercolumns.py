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
def itercolumns(self, *args, **kwargs):
    """
        Iterator over all columns in their numerical order.

        Yields:
            `pyarrow.ChunkedArray`
        """
    return self.table.itercolumns(*args, **kwargs)