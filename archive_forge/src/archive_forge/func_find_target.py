from __future__ import annotations
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstPrinter
from mesonbuild.mesonlib import MesonException, setup_vsenv
from . import mlog, environment
from functools import wraps
from .mparser import Token, ArrayNode, ArgumentNode, AssignmentNode, BaseStringNode, BooleanNode, ElementaryNode, IdNode, FunctionNode, StringNode, SymbolNode
import json, os, re, sys
import typing as T
def find_target(self, target: str):

    def check_list(name: str) -> T.List[BaseNode]:
        result = []
        for i in self.interpreter.targets:
            if name in {i['name'], i['id']}:
                result += [i]
        return result
    targets = check_list(target)
    if targets:
        if len(targets) == 1:
            return targets[0]
        else:
            mlog.error('There are multiple targets matching', mlog.bold(target))
            for i in targets:
                mlog.error('  -- Target name', mlog.bold(i['name']), 'with ID', mlog.bold(i['id']))
            mlog.error('Please try again with the unique ID of the target', *self.on_error())
            self.handle_error()
            return None
    tgt = None
    if target in self.interpreter.assignments:
        node = self.interpreter.assignments[target]
        if isinstance(node, FunctionNode):
            if node.func_name.value in {'executable', 'jar', 'library', 'shared_library', 'shared_module', 'static_library', 'both_libraries'}:
                tgt = self.interpreter.assign_vals[target]
    return tgt