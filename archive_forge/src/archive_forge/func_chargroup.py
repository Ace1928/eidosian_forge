from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
def chargroup(self):
    if self.static_b('^'):
        negate = True
    else:
        negate = False
    groups = []
    while True:
        try:
            groups.append(self.chargroup_inner())
        except nomatch:
            break
    self.static(']')
    if len(groups) == 1:
        f = tuple(groups)[0]
        return _CharGroup(f.chars, negate ^ f.negated)
    elif len(groups) == 0:
        return _CharGroup(frozenset({}), negate)
    else:
        return _combine_char_groups(*groups, negate=negate)