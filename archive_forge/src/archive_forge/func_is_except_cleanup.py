import ast
import sys
import dis
from types import CodeType, FrameType
from typing import Any, Callable, Iterator, Optional, Sequence, Set, Tuple, Type, Union, cast
from .executing import EnhancedAST, NotOneValueFound, Source, only, function_node_types, assert_
from ._exceptions import KnownIssue, VerifierFailure
from functools import lru_cache
@staticmethod
def is_except_cleanup(inst: dis.Instruction, node: EnhancedAST) -> bool:
    if inst.opname not in ('STORE_NAME', 'STORE_FAST', 'STORE_DEREF', 'STORE_GLOBAL', 'DELETE_NAME', 'DELETE_FAST', 'DELETE_DEREF', 'DELETE_GLOBAL'):
        return False
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and inst.opname.startswith('STORE_') and (mangled_name(node) == inst.argval):
        return False
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Del) and inst.opname.startswith('DELETE_') and (mangled_name(node) == inst.argval):
        return False
    return any((isinstance(n, ast.ExceptHandler) and n.name and (mangled_name(n) == inst.argval) for n in parents(node)))