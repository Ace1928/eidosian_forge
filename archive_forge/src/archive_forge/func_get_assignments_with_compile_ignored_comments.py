import contextlib
import copy
import hashlib
import inspect
import io
import pickle
import tokenize
import unittest
import warnings
from types import FunctionType, ModuleType
from typing import Any, Dict, Optional, Set, Tuple, Union
from unittest import mock
def get_assignments_with_compile_ignored_comments(module):
    source_code = inspect.getsource(module)
    assignments = set()
    tokens = tokenize.tokenize(io.BytesIO(source_code.encode('utf-8')).readline)
    current_comment = ('', -1)
    prev_name = ''
    prev_assigned = ('', -1)
    for token in tokens:
        if token.type == tokenize.COMMENT:
            maybe_current = token.string.strip()
            if COMPILE_IGNORED_MARKER in maybe_current:
                assert current_comment == ('', -1), f'unconsumed {COMPILE_IGNORED_MARKER}'
                current_comment = (maybe_current, token.start[0])
                if token.start[0] == prev_assigned[1]:
                    assignments.add(prev_assigned[0])
                    current_comment = ('', -1)
        elif token.type == tokenize.NAME:
            prev_name = token.string
        elif token.type == tokenize.OP and token.string == '=':
            prev_assigned = (prev_name, token.start[0])
            if COMPILE_IGNORED_MARKER in current_comment[0] and current_comment[1] == token.start[0] - 1:
                assignments.add(prev_name)
                current_comment = ('', -1)
    assert current_comment == ('', -1), f'unconsumed {COMPILE_IGNORED_MARKER}'
    return assignments