import __future__
import ast
import dis
import inspect
import io
import linecache
import re
import sys
import types
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from itertools import islice
from itertools import zip_longest
from operator import attrgetter
from pathlib import Path
from threading import RLock
from tokenize import detect_encoding
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Sized, Tuple, \
@cache
def statements_at_line(self, lineno):
    """
        Returns the statement nodes overlapping the given line.

        Returns at most one statement unless semicolons are present.

        If the `text` attribute is not valid python, meaning
        `tree` is None, returns an empty set.

        Otherwise, `Source.for_frame(frame).statements_at_line(frame.f_lineno)`
        should return at least one statement.
        """
    return {statement_containing_node(node) for node in self._nodes_by_line[lineno]}