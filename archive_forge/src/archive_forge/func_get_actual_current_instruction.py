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
def get_actual_current_instruction(self, lasti):
    """
        Get the instruction corresponding to the current
        frame offset, skipping EXTENDED_ARG instructions
        """
    instructions = list(get_instructions(self.code))
    index = only((i for i, inst in enumerate(instructions) if inst.offset == lasti))
    while True:
        instruction = instructions[index]
        if instruction.opname != 'EXTENDED_ARG':
            return instruction
        index += 1