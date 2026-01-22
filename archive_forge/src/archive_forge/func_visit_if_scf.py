import ast
import inspect
import re
import sys
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from .. import language
from .._C.libtriton.triton import ir
from ..language import constexpr, tensor
from ..runtime import JITFunction
from .errors import (CompilationError, CompileTimeAssertionFailure, UnsupportedLanguageConstruct)
def visit_if_scf(self, cond, node):
    with enter_sub_region(self) as sr:
        liveins, _ = sr
        ip, last_loc = self._get_insertion_point_and_loc()
        then_block = self.builder.create_block()
        else_block = self.builder.create_block() if node.orelse else None
        then_defs, else_defs, then_block, else_block, names, ret_types, _ = self.visit_then_else_blocks(node, liveins, then_block, else_block)
        self._set_insertion_point_and_loc(ip, last_loc)
        if_op = self.builder.create_if_op([ty.to_ir(self.builder) for ty in ret_types], cond.handle, True)
        then_block.merge_block_before(if_op.get_then_block())
        self.builder.set_insertion_point_to_end(if_op.get_then_block())
        if len(names) > 0:
            self.builder.create_yield_op([then_defs[n].handle for n in names])
        if not node.orelse:
            else_block = if_op.get_else_block()
        else:
            else_block.merge_block_before(if_op.get_else_block())
        self.builder.set_insertion_point_to_end(if_op.get_else_block())
        if len(names) > 0:
            self.builder.create_yield_op([else_defs[n].handle for n in names])
    for i, name in enumerate(names):
        new_tensor = language.core.tensor(if_op.get_result(i), ret_types[i])
        self.set_value(name, new_tensor)