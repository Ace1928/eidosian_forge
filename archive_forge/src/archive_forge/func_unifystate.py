from ast import literal_eval
from typing import TypeVar, Generic, Mapping, Sequence, Set, Union
from parso.pgen2.grammar_parser import GrammarParser, NFAState
def unifystate(self, old, new):
    for label, next_ in self.arcs.items():
        if next_ is old:
            self.arcs[label] = new