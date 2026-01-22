from __future__ import annotations
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstPrinter
from mesonbuild.mesonlib import MesonException, setup_vsenv
from . import mlog, environment
from functools import wraps
from .mparser import Token, ArrayNode, ArgumentNode, AssignmentNode, BaseStringNode, BooleanNode, ElementaryNode, IdNode, FunctionNode, StringNode, SymbolNode
import json, os, re, sys
import typing as T
@RequiredKeys(rewriter_keys['kwargs'])
def process_kwargs(self, cmd):
    mlog.log('Processing function type', mlog.bold(cmd['function']), 'with id', mlog.cyan("'" + cmd['id'] + "'"))
    if cmd['function'] not in rewriter_func_kwargs:
        mlog.error('Unknown function type', cmd['function'], *self.on_error())
        return self.handle_error()
    kwargs_def = rewriter_func_kwargs[cmd['function']]
    node = None
    arg_node = None
    if cmd['function'] == 'project':
        if {'/', '//'}.isdisjoint({cmd['id']}):
            mlog.error('The ID for the function type project must be "/" or "//" not "' + cmd['id'] + '"', *self.on_error())
            return self.handle_error()
        node = self.interpreter.project_node
        arg_node = node.args
    elif cmd['function'] == 'target':
        tmp = self.find_target(cmd['id'])
        if tmp:
            node = tmp['node']
            arg_node = node.args
    elif cmd['function'] == 'dependency':
        tmp = self.find_dependency(cmd['id'])
        if tmp:
            node = tmp['node']
            arg_node = node.args
    if not node:
        mlog.error('Unable to find the function node')
    assert isinstance(node, FunctionNode)
    assert isinstance(arg_node, ArgumentNode)
    arg_node.kwargs = {k.value: v for k, v in arg_node.kwargs.items()}
    if cmd['operation'] == 'info':
        info_data = {}
        for key, val in sorted(arg_node.kwargs.items()):
            info_data[key] = None
            if isinstance(val, ElementaryNode):
                info_data[key] = val.value
            elif isinstance(val, ArrayNode):
                data_list = []
                for i in val.args.arguments:
                    element = None
                    if isinstance(i, ElementaryNode):
                        element = i.value
                    data_list += [element]
                info_data[key] = data_list
        self.add_info('kwargs', '{}#{}'.format(cmd['function'], cmd['id']), info_data)
        return
    num_changed = 0
    for key, val in sorted(cmd['kwargs'].items()):
        if key not in kwargs_def:
            mlog.error('Cannot modify unknown kwarg', mlog.bold(key), *self.on_error())
            self.handle_error()
            continue
        if cmd['operation'] == 'delete':
            if key not in arg_node.kwargs:
                mlog.log('  -- Key', mlog.bold(key), 'is already deleted')
                continue
            mlog.log('  -- Deleting', mlog.bold(key), 'from the kwargs')
            del arg_node.kwargs[key]
        elif cmd['operation'] == 'set':
            mlog.log('  -- Setting', mlog.bold(key), 'to', mlog.yellow(str(val)))
            arg_node.kwargs[key] = kwargs_def[key].new_node(val)
        else:
            if key not in arg_node.kwargs:
                arg_node.kwargs[key] = None
            modifier = kwargs_def[key](arg_node.kwargs[key])
            if not modifier.can_modify():
                mlog.log('  -- Skipping', mlog.bold(key), 'because it is too complex to modify')
                continue
            val_str = str(val)
            if cmd['operation'] == 'add':
                mlog.log('  -- Adding', mlog.yellow(val_str), 'to', mlog.bold(key))
                modifier.add_value(val)
            elif cmd['operation'] == 'remove':
                mlog.log('  -- Removing', mlog.yellow(val_str), 'from', mlog.bold(key))
                modifier.remove_value(val)
            elif cmd['operation'] == 'remove_regex':
                mlog.log('  -- Removing all values matching', mlog.yellow(val_str), 'from', mlog.bold(key))
                modifier.remove_regex(val)
            arg_node.kwargs[key] = modifier.get_node()
        num_changed += 1
    arg_node.kwargs = {IdNode(Token('', '', 0, 0, 0, None, k)): v for k, v in arg_node.kwargs.items()}
    for k, v in arg_node.kwargs.items():
        k.level = v.level
    if num_changed > 0 and node not in self.modified_nodes:
        self.modified_nodes += [node]