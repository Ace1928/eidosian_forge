import argparse
import builtins
import enum
import importlib
import inspect
import io
import logging
import os
import pickle
import ply.lex
import re
import sys
import textwrap
import types
from operator import attrgetter
from pyomo.common.collections import Sequence, Mapping
from pyomo.common.deprecation import (
from pyomo.common.fileutils import import_file
from pyomo.common.formatting import wrap_reStructuredText
from pyomo.common.modeling import NOTSET
def initialize_argparse(self, parser):

    def _get_subparser_or_group(_parser, name):
        if isinstance(name, argparse._ActionsContainer):
            return (2, name)
        if not isinstance(name, str):
            raise RuntimeError('Unknown datatype (%s) for argparse group on configuration definition %s' % (type(name).__name__, obj.name(True)))
        try:
            for _grp in _parser._subparsers._group_actions:
                if name in _grp._name_parser_map:
                    return (1, _grp._name_parser_map[name])
        except AttributeError:
            pass
        for _grp in _parser._action_groups:
            if _grp.title == name:
                return (0, _grp)
        return (0, _parser.add_argument_group(title=name))

    def _process_argparse_def(obj, _args, _kwds):
        _parser = parser
        _kwds = dict(_kwds)
        if 'group' in _kwds:
            _group = _kwds.pop('group')
            if isinstance(_group, tuple):
                for _idx, _grp in enumerate(_group):
                    _issub, _parser = _get_subparser_or_group(_parser, _grp)
                    if not _issub and _idx < len(_group) - 1:
                        raise RuntimeError("Could not find argparse subparser '%s' for Config item %s" % (_grp, obj.name(True)))
            else:
                _issub, _parser = _get_subparser_or_group(_parser, _group)
        if 'dest' not in _kwds:
            _kwds['dest'] = 'CONFIGBLOCK.' + obj.name(True)
            if 'metavar' not in _kwds and _kwds.get('action', '') not in _store_bool and (obj._domain is not None):
                _kwds['metavar'] = obj.domain_name().upper()
        _parser.add_argument(*_args, default=argparse.SUPPRESS, **_kwds)
    for level, prefix, value, obj in self._data_collector(None, ''):
        if obj._argparse is None:
            continue
        for _args, _kwds in obj._argparse:
            _process_argparse_def(obj, _args, _kwds)