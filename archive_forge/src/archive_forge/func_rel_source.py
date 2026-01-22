from __future__ import annotations
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstPrinter
from mesonbuild.mesonlib import MesonException, setup_vsenv
from . import mlog, environment
from functools import wraps
from .mparser import Token, ArrayNode, ArgumentNode, AssignmentNode, BaseStringNode, BooleanNode, ElementaryNode, IdNode, FunctionNode, StringNode, SymbolNode
import json, os, re, sys
import typing as T
def rel_source(src: str) -> str:
    subdir = os.path.abspath(os.path.join(self.sourcedir, target['subdir']))
    if os.path.isabs(src):
        return os.path.relpath(src, subdir)
    elif not os.path.exists(src):
        return src
    return os.path.relpath(os.path.abspath(src), subdir)