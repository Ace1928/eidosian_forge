from __future__ import annotations
from collections import deque
from dask.core import istask, subs
def iter_matches(self, term):
    """A generator that lazily finds matchings for term from the RuleSet.

        Parameters
        ----------
        term : task

        Yields
        ------
        Tuples of `(rule, subs)`, where `rule` is the rewrite rule being
        matched, and `subs` is a dictionary mapping the variables in the lhs
        of the rule to their matching values in the term."""
    S = Traverser(term)
    for m, syms in _match(S, self._net):
        for i in m:
            rule = self.rules[i]
            subs = _process_match(rule, syms)
            if subs is not None:
                yield (rule, subs)