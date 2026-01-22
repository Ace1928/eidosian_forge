from typing import Dict, Set, Iterator, Tuple, List, TypeVar, Generic
from collections import defaultdict
from ..utils import classify, classify_bool, bfs, fzset, Enumerator, logger
from ..exceptions import GrammarError
from .grammar_analysis import GrammarAnalyzer, Terminal, LR0ItemSet, RulePtr, State
from ..grammar import Rule, Symbol
from ..common import ParserConf
def compute_reads_relations(self):
    for root in self.lr0_start_states.values():
        assert len(root.kernel) == 1
        for rp in root.kernel:
            assert rp.index == 0
            self.directly_reads[root, rp.next] = set([Terminal('$END')])
    for state in self.lr0_itemsets:
        seen = set()
        for rp in state.closure:
            if rp.is_satisfied:
                continue
            s = rp.next
            if s not in self.lr0_rules_by_origin:
                continue
            if s in seen:
                continue
            seen.add(s)
            nt = (state, s)
            self.nonterminal_transitions.append(nt)
            dr = self.directly_reads[nt]
            r = self.reads[nt]
            next_state = state.transitions[s]
            for rp2 in next_state.closure:
                if rp2.is_satisfied:
                    continue
                s2 = rp2.next
                if s2 not in self.lr0_rules_by_origin:
                    dr.add(s2)
                if s2 in self.NULLABLE:
                    r.add((next_state, s2))