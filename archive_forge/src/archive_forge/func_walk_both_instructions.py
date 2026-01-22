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
def walk_both_instructions(original_instructions, original_start, instructions, start):
    """
    Yields matching indices and instructions from the new and original instructions,
    leaving out changes made by the sentinel transformation.
    """
    original_iter = islice(enumerate(original_instructions), original_start, None)
    new_iter = non_sentinel_instructions(instructions, start)
    inverted_comparison = False
    while True:
        try:
            original_i, original_inst = next(original_iter)
            new_i, new_inst = next(new_iter)
        except StopIteration:
            return
        if inverted_comparison and original_inst.opname != new_inst.opname == 'UNARY_NOT':
            new_i, new_inst = next(new_iter)
        inverted_comparison = original_inst.opname == new_inst.opname in ('CONTAINS_OP', 'IS_OP') and original_inst.arg != new_inst.arg
        yield (original_i, original_inst, new_i, new_inst)