from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def parse_inline_table(src: str, pos: Pos, parse_float: ParseFloat) -> tuple[Pos, dict]:
    pos += 1
    nested_dict = NestedDict()
    flags = Flags()
    pos = skip_chars(src, pos, TOML_WS)
    if src.startswith('}', pos):
        return (pos + 1, nested_dict.dict)
    while True:
        pos, key, value = parse_key_value_pair(src, pos, parse_float)
        key_parent, key_stem = (key[:-1], key[-1])
        if flags.is_(key, Flags.FROZEN):
            raise suffixed_err(src, pos, f'Cannot mutate immutable namespace {key}')
        try:
            nest = nested_dict.get_or_create_nest(key_parent, access_lists=False)
        except KeyError:
            raise suffixed_err(src, pos, 'Cannot overwrite a value') from None
        if key_stem in nest:
            raise suffixed_err(src, pos, f'Duplicate inline table key {key_stem!r}')
        nest[key_stem] = value
        pos = skip_chars(src, pos, TOML_WS)
        c = src[pos:pos + 1]
        if c == '}':
            return (pos + 1, nested_dict.dict)
        if c != ',':
            raise suffixed_err(src, pos, 'Unclosed inline table')
        if isinstance(value, (dict, list)):
            flags.set(key, Flags.FROZEN, recursive=True)
        pos += 1
        pos = skip_chars(src, pos, TOML_WS)