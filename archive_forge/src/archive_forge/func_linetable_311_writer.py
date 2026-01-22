import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def linetable_311_writer(first_lineno: int):
    """
    Used to create typing.CodeType.co_linetable
    See https://github.com/python/cpython/blob/3.11/Objects/locations.md
    This is the internal format of the line number table for Python 3.11
    """
    assert sys.version_info >= (3, 11)
    linetable = []
    lineno = first_lineno

    def update(positions: 'dis.Positions', inst_size):
        nonlocal lineno
        lineno_new = positions.lineno if positions else None

        def _update(delta, size):
            assert 0 < size <= 8
            other_varints: Tuple[int, ...] = ()
            if positions and positions.lineno is not None and (positions.end_lineno is not None) and (positions.col_offset is not None) and (positions.end_col_offset is not None):
                linetable.append(240 + size - 1)
                other_varints = (positions.end_lineno - positions.lineno, positions.col_offset + 1, positions.end_col_offset + 1)
            else:
                linetable.append(232 + size - 1)
            if delta < 0:
                delta = -delta << 1 | 1
            else:
                delta <<= 1
            linetable.extend(encode_varint(delta))
            for n in other_varints:
                linetable.extend(encode_varint(n))
        if lineno_new is None:
            lineno_delta = 0
        else:
            lineno_delta = lineno_new - lineno
            lineno = lineno_new
        while inst_size > 8:
            _update(lineno_delta, 8)
            inst_size -= 8
        _update(lineno_delta, inst_size)
    return (linetable, update)