from typing import Dict, Set, Iterator, Tuple, List, TypeVar, Generic
from collections import defaultdict
from ..utils import classify, classify_bool, bfs, fzset, Enumerator, logger
from ..exceptions import GrammarError
from .grammar_analysis import GrammarAnalyzer, Terminal, LR0ItemSet, RulePtr, State
from ..grammar import Rule, Symbol
from ..common import ParserConf
def compute_lalr1_states(self) -> None:
    m: Dict[LR0ItemSet, Dict[str, Tuple]] = {}
    reduce_reduce = []
    for itemset in self.lr0_itemsets:
        actions: Dict[Symbol, Tuple] = {la: (Shift, next_state.closure) for la, next_state in itemset.transitions.items()}
        for la, rules in itemset.lookaheads.items():
            if len(rules) > 1:
                p = [(r.options.priority or 0, r) for r in rules]
                p.sort(key=lambda r: r[0], reverse=True)
                best, second_best = p[:2]
                if best[0] > second_best[0]:
                    rules = {best[1]}
                else:
                    reduce_reduce.append((itemset, la, rules))
                    continue
            rule, = rules
            if la in actions:
                if self.strict:
                    raise GrammarError(f'Shift/Reduce conflict for terminal {la.name}. [strict-mode]\n ')
                elif self.debug:
                    logger.warning('Shift/Reduce conflict for terminal %s: (resolving as shift)', la.name)
                    logger.warning(' * %s', rule)
                else:
                    logger.debug('Shift/Reduce conflict for terminal %s: (resolving as shift)', la.name)
                    logger.debug(' * %s', rule)
            else:
                actions[la] = (Reduce, rule)
        m[itemset] = {k.name: v for k, v in actions.items()}
    if reduce_reduce:
        msgs = []
        for itemset, la, rules in reduce_reduce:
            msg = 'Reduce/Reduce collision in %s between the following rules: %s' % (la, ''.join(['\n\t- ' + str(r) for r in rules]))
            if self.debug:
                msg += '\n    collision occurred in state: {%s\n    }' % ''.join(['\n\t' + str(x) for x in itemset.closure])
            msgs.append(msg)
        raise GrammarError('\n\n'.join(msgs))
    states = {k.closure: v for k, v in m.items()}
    end_states: Dict[str, 'State'] = {}
    for state in states:
        for rp in state:
            for start in self.lr0_start_states:
                if rp.rule.origin.name == '$root_' + start and rp.is_satisfied:
                    assert start not in end_states
                    end_states[start] = state
    start_states = {start: state.closure for start, state in self.lr0_start_states.items()}
    _parse_table = ParseTable(states, start_states, end_states)
    if self.debug:
        self.parse_table = _parse_table
    else:
        self.parse_table = IntParseTable.from_ParseTable(_parse_table)