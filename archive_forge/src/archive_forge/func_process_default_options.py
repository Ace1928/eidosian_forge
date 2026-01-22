from __future__ import annotations
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstPrinter
from mesonbuild.mesonlib import MesonException, setup_vsenv
from . import mlog, environment
from functools import wraps
from .mparser import Token, ArrayNode, ArgumentNode, AssignmentNode, BaseStringNode, BooleanNode, ElementaryNode, IdNode, FunctionNode, StringNode, SymbolNode
import json, os, re, sys
import typing as T
@RequiredKeys(rewriter_keys['default_options'])
def process_default_options(self, cmd):
    kwargs_cmd = {'function': 'project', 'id': '/', 'operation': 'remove_regex', 'kwargs': {'default_options': [f'{x}=.*' for x in cmd['options'].keys()]}}
    self.process_kwargs(kwargs_cmd)
    if cmd['operation'] != 'set':
        return
    kwargs_cmd['operation'] = 'add'
    kwargs_cmd['kwargs']['default_options'] = []
    cdata = self.interpreter.coredata
    options = {**{str(k): v for k, v in cdata.options.items()}, **{str(k): v for k, v in cdata.options.items()}, **{str(k): v for k, v in cdata.options.items()}, **{str(k): v for k, v in cdata.options.items()}, **{str(k): v for k, v in cdata.options.items()}}
    for key, val in sorted(cmd['options'].items()):
        if key not in options:
            mlog.error('Unknown options', mlog.bold(key), *self.on_error())
            self.handle_error()
            continue
        try:
            val = options[key].validate_value(val)
        except MesonException as e:
            mlog.error('Unable to set', mlog.bold(key), mlog.red(str(e)), *self.on_error())
            self.handle_error()
            continue
        kwargs_cmd['kwargs']['default_options'] += [f'{key}={val}']
    self.process_kwargs(kwargs_cmd)