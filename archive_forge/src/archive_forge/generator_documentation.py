from ast import literal_eval
from typing import TypeVar, Generic, Mapping, Sequence, Set, Union
from parso.pgen2.grammar_parser import GrammarParser, NFAState

    Calculates the first plan in the first_plans dictionary for every given
    nonterminal. This is going to be used to know when to create stack nodes.
    