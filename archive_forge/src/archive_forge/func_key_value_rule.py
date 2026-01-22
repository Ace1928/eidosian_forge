from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def key_value_rule(src: str, pos: Pos, out: Output, header: Key, parse_float: ParseFloat) -> Pos:
    pos, key, value = parse_key_value_pair(src, pos, parse_float)
    key_parent, key_stem = (key[:-1], key[-1])
    abs_key_parent = header + key_parent
    relative_path_cont_keys = (header + key[:i] for i in range(1, len(key)))
    for cont_key in relative_path_cont_keys:
        if out.flags.is_(cont_key, Flags.EXPLICIT_NEST):
            raise suffixed_err(src, pos, f'Cannot redefine namespace {cont_key}')
        out.flags.add_pending(cont_key, Flags.EXPLICIT_NEST)
    if out.flags.is_(abs_key_parent, Flags.FROZEN):
        raise suffixed_err(src, pos, f'Cannot mutate immutable namespace {abs_key_parent}')
    try:
        nest = out.data.get_or_create_nest(abs_key_parent)
    except KeyError:
        raise suffixed_err(src, pos, 'Cannot overwrite a value') from None
    if key_stem in nest:
        raise suffixed_err(src, pos, 'Cannot overwrite a value')
    if isinstance(value, (dict, list)):
        out.flags.set(header + key, Flags.FROZEN, recursive=True)
    nest[key_stem] = value
    return pos