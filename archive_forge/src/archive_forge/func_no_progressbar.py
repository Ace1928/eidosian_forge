import json
import sys
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterable, Iterator
import ase.io
from ase.db import connect
from ase.db.core import convert_str_to_int_float_or_str
from ase.db.row import row2dct
from ase.db.table import Table, all_columns
from ase.utils import plural
@contextmanager
def no_progressbar(iterable: Iterable, length: int=None) -> Iterator[Iterable]:
    """A do-nothing implementation."""
    yield iterable