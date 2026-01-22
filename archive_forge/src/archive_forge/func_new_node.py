from __future__ import annotations
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstPrinter
from mesonbuild.mesonlib import MesonException, setup_vsenv
from . import mlog, environment
from functools import wraps
from .mparser import Token, ArrayNode, ArgumentNode, AssignmentNode, BaseStringNode, BooleanNode, ElementaryNode, IdNode, FunctionNode, StringNode, SymbolNode
import json, os, re, sys
import typing as T
@classmethod
def new_node(cls, value=None):
    if value is None:
        value = []
    elif not isinstance(value, list):
        return cls._new_element_node(value)
    args = ArgumentNode(Token('', '', 0, 0, 0, None, ''))
    args.arguments = [cls._new_element_node(i) for i in value]
    return ArrayNode(_symbol('['), args, _symbol(']'))