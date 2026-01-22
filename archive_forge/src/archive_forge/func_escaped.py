from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
def escaped(self, inner=False):
    if self.static_b('x'):
        n = self.multiple('0123456789abcdefABCDEF', 2, 2)
        c = chr(int(n, 16))
        return _CharGroup(frozenset({c}), False)
    if self.static_b('0'):
        n = self.multiple('01234567', 1, 2)
        c = chr(int(n, 8))
        return _CharGroup(frozenset({c}), False)
    if self.anyof_b('N', 'p', 'P', 'u', 'U'):
        raise Unsupported('regex module unicode properties are not supported.')
    if not inner:
        try:
            n = self.multiple('01234567', 3, 3)
        except nomatch:
            pass
        else:
            c = chr(int(n, 8))
            return _CharGroup(frozenset({c}), False)
        try:
            self.multiple('0123456789', 1, 2)
        except nomatch:
            pass
        else:
            raise Unsupported('Group references are not implemented')
    else:
        try:
            n = self.multiple('01234567', 1, 3)
        except nomatch:
            pass
        else:
            c = chr(int(n, 8))
            return _CharGroup(frozenset({c}), False)
    if not inner:
        try:
            c = self.anyof(*self.RESERVED_ESCAPES)
        except nomatch:
            pass
        else:
            raise Unsupported(f'Escape \\{c} is not implemented')
    try:
        c = self.anyof(*_CHAR_GROUPS)
    except nomatch:
        pass
    else:
        return _CHAR_GROUPS[c]
    c = self.any_but('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    if c.isalpha():
        raise nomatch
    return _CharGroup(frozenset(c), False)