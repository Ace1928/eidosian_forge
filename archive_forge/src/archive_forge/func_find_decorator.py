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
def find_decorator(self, stmts):
    stmt = only(stmts)
    assert_(isinstance(stmt, (ast.ClassDef, function_node_types)))
    decorators = stmt.decorator_list
    assert_(decorators)
    line_instructions = [inst for inst in self.clean_instructions(self.code) if inst.lineno == self.frame.f_lineno]
    last_decorator_instruction_index = [i for i, inst in enumerate(line_instructions) if inst.opname == 'CALL_FUNCTION'][-1]
    assert_(line_instructions[last_decorator_instruction_index + 1].opname.startswith('STORE_'))
    decorator_instructions = line_instructions[last_decorator_instruction_index - len(decorators) + 1:last_decorator_instruction_index + 1]
    assert_({inst.opname for inst in decorator_instructions} == {'CALL_FUNCTION'})
    decorator_index = decorator_instructions.index(self.instruction)
    decorator = decorators[::-1][decorator_index]
    self.decorator = decorator
    self.result = stmt