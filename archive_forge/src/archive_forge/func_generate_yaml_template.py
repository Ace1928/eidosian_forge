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
def generate_yaml_template(self, indent_spacing=2, width=78, visibility=0):
    minDocWidth = 20
    comment = '  # '
    data = list(self._data_collector(0, '', visibility))
    level_info = {}
    for lvl, pre, val, obj in data:
        _str = _value2yaml(pre, val, obj)
        if lvl not in level_info:
            level_info[lvl] = {'data': [], 'off': 0, 'line': 0, 'over': 0}
        level_info[lvl]['data'].append((_str.find(':') + 2, len(_str), len(obj._description or '')))
    for lvl in sorted(level_info):
        indent = lvl * indent_spacing
        _ok = width - indent - len(comment) - minDocWidth
        offset = max((val if val < _ok else key for key, val, doc in level_info[lvl]['data']))
        offset += indent + len(comment)
        over = sum((1 for key, val, doc in level_info[lvl]['data'] if doc + offset > width))
        if len(level_info[lvl]['data']) - over > 0:
            line = max((offset + doc for key, val, doc in level_info[lvl]['data'] if offset + doc <= width))
        else:
            line = width
        level_info[lvl]['off'] = offset
        level_info[lvl]['line'] = line
        level_info[lvl]['over'] = over
    maxLvl = 0
    maxDoc = 0
    pad = 0
    for lvl in sorted(level_info):
        _pad = level_info[lvl]['off']
        _doc = level_info[lvl]['line'] - _pad
        if _pad > pad:
            if maxDoc + _pad <= width:
                pad = _pad
            else:
                break
        if _doc + pad > width:
            break
        if _doc > maxDoc:
            maxDoc = _doc
        maxLvl = lvl
    os = io.StringIO()
    if self._description:
        os.write(comment.lstrip() + self._description + '\n')
    for lvl, pre, val, obj in data:
        _str = _value2yaml(pre, val, obj)
        if not obj._description:
            os.write(' ' * indent_spacing * lvl + _str + '\n')
            continue
        if lvl <= maxLvl:
            field = pad - len(comment)
        else:
            field = level_info[lvl]['off'] - len(comment)
        os.write(' ' * indent_spacing * lvl)
        if width - len(_str) - minDocWidth >= 0:
            os.write('%%-%ds' % (field - indent_spacing * lvl) % _str)
        else:
            os.write(_str + '\n' + ' ' * field)
        os.write(comment)
        txtArea = max(width - field - len(comment), minDocWidth)
        os.write(('\n' + ' ' * field + comment).join(textwrap.wrap(obj._description, txtArea, subsequent_indent='  ')))
        os.write('\n')
    return os.getvalue()