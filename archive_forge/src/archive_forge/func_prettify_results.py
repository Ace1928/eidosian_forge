import builtins
import collections
import functools
import inspect
import itertools
import logging
import math
import operator
import re
import sys
import threading
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, cast, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, Iterable
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx.experimental import _config as config
from torch.fx.experimental.recording import (
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch import SymBool, SymFloat, SymInt
from torch._guards import ShapeGuard, Source, TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import FloorDiv, Mod, IsNonOverlappingAndDenseIndicator
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.value_ranges import bound_sympy, SymPyValueRangeAnalysis, ValueRanges, ValueRangeError
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._traceback import format_frame, CapturedTraceback
from torch._utils_internal import signpost_event
from torch._logging import LazyString
import sympy
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
def prettify_results(self, original_signature: inspect.Signature, constraint_violation_error=None, forced_specializations=None):
    if self._dcp.source_name_to_debug_name:

        def transform(s):
            for k, v in self._dcp.source_name_to_debug_name.items():
                s = s.replace(k, v)
            return s
        results = defaultdict(dict)

        def flip(op):
            if op == '<=':
                return '>='
            if op == '>=':
                return '<='
            if op == '<':
                return '>'
            if op == '>':
                return '<'
            assert op == '=='
            return op

        def relation_with_digit(expr, op, digit):
            if op == '<=':
                results[expr]['max'] = digit
            elif op == '<':
                results[expr]['max'] = digit - 1
            elif op == '>=':
                results[expr]['min'] = digit
            elif op == '>':
                results[expr]['min'] = digit + 1
            else:
                assert op == '=='
                results[expr]['eq'] = digit
        for s in self._static_results.union(self._dynamic_results):
            t = transform(s)
            if t == s:
                continue
            left, op, right = t.split(' ')
            if op == '==' and left == right:
                continue
            if right.isdigit():
                relation_with_digit(left, op, int(right))
            elif left.isdigit():
                relation_with_digit(right, flip(op), int(left))
            else:
                assert op == '=='
                results[left]['eq'] = right
        buf = ''
        debug_names = set()
        if forced_specializations:
            debug_names.update((k.split(' = ')[0] for k in forced_specializations.keys()))
            buf += f'Specializations unexpectedly required ({', '.join(debug_names)})! For more information, run with TORCH_LOGS=dynamic.\n'
            for s, val in forced_specializations.items():
                buf += f'  - {s} must be specialized to {val} because the guards generated for it are too complex.\n'
        dims = []
        others = []
        match = None
        if constraint_violation_error:
            match = re.search('Constraints violated \\((.*)\\)', constraint_violation_error.args[0])
        if match is not None:
            debug_names.update(match.expand('\\1').split(', '))
        for k, c in results.items():
            if k not in debug_names:
                continue
            if 'eq' in c:
                other = c['eq']
                if isinstance(other, int):
                    others.append(f'{k} = None  # {other}')
                else:
                    others.append(f'{k} = {other}')
            else:
                min_ = c.get('min', None)
                if min_ == 2:
                    min_ = None
                max_ = c.get('max', None)
                if min_ is not None and max_ is not None:
                    dims.append(f"{k} = Dim('{k}', min={min_}, max={max_})")
                elif min_ is not None:
                    dims.append(f"{k} = Dim('{k}', min={min_})")
                elif max_ is not None:
                    dims.append(f"{k} = Dim('{k}', max={max_})")
                else:
                    dims.append(f"{k} = Dim('{k}')")
        buf += '\nSuggested fixes:\n  '
        buf += '\n  '.join(dims + others)
        return buf

    def extract_and_rewrite_local(dc):
        match = re.search("L\\['(.+?)'\\]", dc)
        if match is None:
            return
        arg = match.expand('\\1')
        dc = re.sub("L\\['(.+?)'\\]", '\\1', dc)
        return (arg, dc)

    def group(results, args_index):
        groups = defaultdict(list)
        for dc in results:
            local = extract_and_rewrite_local(dc)
            if local is None:
                continue
            arg, dc = local
            if arg in args_index:
                groups[args_index[arg]].append(dc)
            else:
                continue
        sorted_groups = []
        for idx, dcs in sorted(groups.items()):
            _, arg = idx
            sorted_groups.append((arg, sorted(dcs)))
        return sorted_groups
    signature = original_signature.replace(return_annotation=inspect.Signature.empty)
    args_index = {}
    for i, arg in enumerate(signature.parameters.keys()):
        args_index[arg] = (i, arg)

    def print_results(grouped, indent, result_fn):
        nonlocal buf
        space = False
        for arg, results in grouped:
            if space:
                buf += '\n'
            else:
                space = True
            buf += f'\n{indent}# {arg}:'
            for result in results:
                buf += f'\n{indent}{result_fn(result)}'
    buf = ''
    if forced_specializations:
        buf += 'Some dynamic dimensions need to be specialized because the constraints inferred for them are too complex to specify.\n'
        for s, val in forced_specializations.items():
            buf += f'  - {s}, which was marked dynamic, must be specialized to {val}.\n'
    indent = 4 * ' '
    if self._static_results:
        grouped_static_results = group(self._static_results, args_index)
        buf += '\nThe following dimensions have been specialized and CANNOT be dynamic.'
        buf += f'\n```\ndef specializations{str(signature)}:'
        print_results(grouped_static_results, indent, lambda result: f'assert {result}')
        buf += '\n```\n'
    if self._dynamic_results:
        grouped_dynamic_results = group(self._dynamic_results, args_index)
        buf += '\nThe following dimensions CAN be dynamic.'
        buf += '\nPlease use the following code to specify the constraints they must satisfy:'
        buf += f'\n```\ndef specify_constraints{str(signature)}:'
        buf += f'\n{indent}return ['
        print_results(grouped_dynamic_results, indent * 2, lambda result: f'{result},')
        buf += f'\n{indent}]\n```\n'
    return buf