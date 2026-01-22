from __future__ import annotations
from itertools import count
import logging
from typing import TYPE_CHECKING
from toolz import unique, concat, pluck, get, memoize
from numba import literal_unroll
import numpy as np
import xarray as xr
from .antialias import AntialiasCombination
from .reductions import SpecialColumn, UsesCudaMutex, by, category_codes, summary
from .utils import (isnull, ngjit,
def make_append(bases, cols, calls, glyph, antialias):
    names = ('_{0}'.format(i) for i in count())
    inputs = list(bases) + list(cols)
    namespace = {}
    need_isnull = any((call[3] for call in calls))
    if need_isnull:
        namespace['isnull'] = isnull
    global_cuda_mutex = any((call[6] == UsesCudaMutex.Global for call in calls))
    any_uses_cuda_mutex = any((call[6] != UsesCudaMutex.No for call in calls))
    if any_uses_cuda_mutex:
        inputs += ['_cuda_mutex']
        namespace['cuda_mutex_lock'] = cuda_mutex_lock
        namespace['cuda_mutex_unlock'] = cuda_mutex_unlock
    signature = [next(names) for i in inputs]
    arg_lk = dict(zip(inputs, signature))
    local_lk = {}
    head = []
    body = []
    ndims = glyph.ndims
    if ndims is not None:
        subscript = ', '.join(['i' + str(n) for n in range(ndims)])
    else:
        subscript = None
    prev_local_cuda_mutex = False
    categorical_args = {}
    where_selectors = {}
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'global_cuda_mutex {global_cuda_mutex}')
        logger.debug(f'any_uses_cuda_mutex {any_uses_cuda_mutex}')
        for k, v in arg_lk.items():
            logger.debug(f'arg_lk {v} {k} {getattr(k, 'column', None)}')

    def get_cuda_mutex_call(lock: bool) -> str:
        func = 'cuda_mutex_lock' if lock else 'cuda_mutex_unlock'
        return f'{func}({arg_lk['_cuda_mutex']}, (y, x))'
    for index, (func, bases, cols, nan_check_column, temps, _, uses_cuda_mutex, categorical) in enumerate(calls):
        local_cuda_mutex = not global_cuda_mutex and uses_cuda_mutex == UsesCudaMutex.Local
        local_lk.update(zip(temps, (next(names) for i in temps)))
        func_name = next(names)
        logger.debug(f'func {func_name} {func}')
        namespace[func_name] = func
        args = [arg_lk[i] for i in bases]
        if categorical and isinstance(cols[0], category_codes):
            args.extend(('{0}[{1}]'.format(arg_lk[col], subscript) for col in cols[1:]))
        elif ndims is None:
            args.extend(('{0}'.format(arg_lk[i]) for i in cols))
        elif categorical:
            args.extend(('{0}[{1}][1]'.format(arg_lk[i], subscript) for i in cols))
        else:
            args.extend(('{0}[{1}]'.format(arg_lk[i], subscript) for i in cols))
        if categorical:
            categorical_arg = arg_lk[cols[0]]
            cat_name = categorical_args.get(categorical_arg, None)
            if cat_name is None:
                col_index = '' if isinstance(cols[0], category_codes) else '[0]'
                cat_name = f'cat{next(names)}'
                categorical_args[categorical_arg] = cat_name
                head.append(f'{cat_name} = int({categorical_arg}[{subscript}]{col_index})')
            arg = signature[index]
            head.append(f'{arg} = {arg}[:, :, {cat_name}]')
        args.extend([local_lk[i] for i in temps])
        if antialias:
            args += ['aa_factor', 'prev_aa_factor']
        if local_cuda_mutex and prev_local_cuda_mutex:
            body.pop()
        is_where = len(bases) == 1 and bases[0].is_where()
        if is_where:
            where_reduction = bases[0]
            if isinstance(where_reduction, by):
                where_reduction = where_reduction.reduction
            selector_hash = hash(where_reduction.selector)
            update_index_arg_name = where_selectors.get(selector_hash, None)
            new_selector = update_index_arg_name is None
            if new_selector:
                update_index_arg_name = next(names)
                where_selectors[selector_hash] = update_index_arg_name
            args.append(update_index_arg_name)
            prev_body = body.pop()
            if local_cuda_mutex and (not prev_local_cuda_mutex):
                body.append(get_cuda_mutex_call(True))
            if new_selector:
                body.append(f'{update_index_arg_name} = {prev_body}')
            else:
                body.append(prev_body)
            if nan_check_column is None:
                whitespace = ''
            else:
                var = f'{arg_lk[nan_check_column]}[{subscript}]'
                prev_body = body[-1]
                body[-1] = f'if not isnull({var}):'
                body.append(f'    {prev_body}')
                whitespace = '    '
            body.append(f'{whitespace}if {update_index_arg_name} >= 0:')
            body.append(f'    {whitespace}{func_name}(x, y, {', '.join(args)})')
        else:
            if local_cuda_mutex and (not prev_local_cuda_mutex):
                body.append(get_cuda_mutex_call(True))
            if nan_check_column:
                var = f'{arg_lk[nan_check_column]}[{subscript}]'
                body.append(f'if not isnull({var}):')
                body.append(f'    {func_name}(x, y, {', '.join(args)})')
            else:
                body.append(f'{func_name}(x, y, {', '.join(args)})')
        if local_cuda_mutex:
            body.append(get_cuda_mutex_call(False))
        prev_local_cuda_mutex = local_cuda_mutex
    body = head + ['{0} = {1}[y, x]'.format(name, arg_lk[agg]) for agg, name in local_lk.items()] + body
    if global_cuda_mutex:
        body = [get_cuda_mutex_call(True)] + body + [get_cuda_mutex_call(False)]
    if antialias:
        signature = ['aa_factor', 'prev_aa_factor'] + signature
    if ndims is None:
        code = 'def append(x, y, {0}):\n    {1}'.format(', '.join(signature), '\n    '.join(body))
    else:
        code = 'def append({0}, x, y, {1}):\n    {2}'.format(subscript, ', '.join(signature), '\n    '.join(body))
    logger.debug(code)
    exec(code, namespace)
    return (ngjit(namespace['append']), any_uses_cuda_mutex)