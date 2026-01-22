from typing import Dict, Set, Iterator, Tuple, List, TypeVar, Generic
from collections import defaultdict
from ..utils import classify, classify_bool, bfs, fzset, Enumerator, logger
from ..exceptions import GrammarError
from .grammar_analysis import GrammarAnalyzer, Terminal, LR0ItemSet, RulePtr, State
from ..grammar import Rule, Symbol
from ..common import ParserConf
def compute_lookaheads(self):
    read_sets = digraph(self.nonterminal_transitions, self.reads, self.directly_reads)
    follow_sets = digraph(self.nonterminal_transitions, self.includes, read_sets)
    for nt, lookbacks in self.lookback.items():
        for state, rule in lookbacks:
            for s in follow_sets[nt]:
                state.lookaheads[s].add(rule)