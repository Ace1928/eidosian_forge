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
def visit_While(self, node):
    with enter_sub_region(self) as sr:
        liveins, insert_block = sr
        ip, last_loc = self._get_insertion_point_and_loc()
        dummy = self.builder.create_block()
        self.builder.set_insertion_point_to_start(dummy)
        self.scf_stack.append(node)
        self.visit_compound_statement(node.body)
        self.scf_stack.pop()
        loop_defs = self.local_defs
        dummy.erase()
        names = []
        ret_types = []
        init_args = []
        for name in loop_defs:
            if name in liveins:
                assert _is_triton_tensor(loop_defs[name]), f'cannoe reassign constxpr {name} in the loop'
                assert _is_triton_tensor(liveins[name]), f'cannot reasign constexpr {name} in the loop'
                assert loop_defs[name].type == liveins[name].type, f'Loop-carried variable {name} has initial type {liveins[name].type} but is re-assigned to {loop_defs[name].type} in loop! Please make sure that the type stays consistent.'
                names.append(name)
                ret_types.append(loop_defs[name].type)
                init_args.append(liveins[name])
        self._set_insertion_point_and_loc(ip, last_loc)
        while_op = self.builder.create_while_op([ty.to_ir(self.builder) for ty in ret_types], [arg.handle for arg in init_args])
        before_block = self.builder.create_block_with_parent(while_op.get_before(), [ty.to_ir(self.builder) for ty in ret_types])
        self.builder.set_insertion_point_to_start(before_block)
        for i, name in enumerate(names):
            self.lscope[name] = language.core.tensor(before_block.arg(i), ret_types[i])
            self.local_defs[name] = self.lscope[name]
        cond = self.visit(node.test)
        self.builder.set_insertion_point_to_end(before_block)
        self.builder.create_condition_op(cond.handle, [before_block.arg(i) for i in range(len(init_args))])
        after_block = self.builder.create_block_with_parent(while_op.get_after(), [ty.to_ir(self.builder) for ty in ret_types])
        self.builder.set_insertion_point_to_start(after_block)
        for i, name in enumerate(names):
            self.lscope[name] = language.core.tensor(after_block.arg(i), ret_types[i])
            self.local_defs[name] = self.lscope[name]
        self.scf_stack.append(node)
        self.visit_compound_statement(node.body)
        self.scf_stack.pop()
        loop_defs = self.local_defs
        yields = []
        for name in loop_defs:
            if name in liveins:
                yields.append(loop_defs[name])
        self.builder.create_yield_op([y.handle for y in yields])
    for i, name in enumerate(names):
        new_def = language.core.tensor(while_op.get_result(i), ret_types[i])
        self.lscope[name] = new_def
        self.local_defs[name] = new_def
    for stmt in node.orelse:
        assert False, 'Not implemented'
        ast.NodeVisitor.generic_visit(self, stmt)