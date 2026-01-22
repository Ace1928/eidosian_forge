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
def opnames_match(inst1, inst2):
    return inst1.opname == inst2.opname or ('JUMP' in inst1.opname and 'JUMP' in inst2.opname) or (inst1.opname == 'PRINT_EXPR' and inst2.opname == 'POP_TOP') or (inst1.opname in ('LOAD_METHOD', 'LOOKUP_METHOD') and inst2.opname == 'LOAD_ATTR') or (inst1.opname == 'CALL_METHOD' and inst2.opname == 'CALL_FUNCTION')