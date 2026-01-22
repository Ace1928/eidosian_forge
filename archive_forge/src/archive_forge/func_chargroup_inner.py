from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
def chargroup_inner(self) -> _CharGroup:
    start = self.index
    if self.static_b('\\'):
        base = self.escaped(True)
    else:
        base = _CharGroup(frozenset(self.any_but(*self.SPECIAL_CHARS_INNER)), False)
    if self.static_b('-'):
        if self.static_b('\\'):
            end = self.escaped(True)
        elif self.peek_static(']'):
            return _combine_char_groups(base, _CharGroup(frozenset('-'), False), negate=False)
        else:
            end = _CharGroup(frozenset(self.any_but(*self.SPECIAL_CHARS_INNER)), False)
        if len(base.chars) != 1 or len(end.chars) != 1:
            raise InvalidSyntax(f'Invalid Character-range: {self.data[start:self.index]}')
        low, high = (ord(*base.chars), ord(*end.chars))
        if low > high:
            raise InvalidSyntax(f'Invalid Character-range: {self.data[start:self.index]}')
        return _CharGroup(frozenset((chr(i) for i in range(low, high + 1))), False)
    return base