from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
def strings(self, max_iterations=None):
    """
            Generate strings (lists of symbols) that this FSM accepts. Since there may
            be infinitely many of these we use a generator instead of constructing a
            static list. Strings will be sorted in order of length and then lexically.
            This procedure uses arbitrary amounts of memory but is very fast. There
            may be more efficient ways to do this, that I haven't investigated yet.
            You can use this in list comprehensions.

            `max_iterations` controls how many attempts will be made to generate strings.
            For complex FSM it can take minutes to actually find something.
            If this isn't acceptable, provide a value to `max_iterations`.
            The approximate time complexity is
            0.15 seconds per 10_000 iterations per 10 symbols
        """
    livestates = set((state for state in self.states if self.islive(state)))
    strings = deque()
    cstate = self.initial
    cstring = []
    if cstate in livestates:
        if cstate in self.finals:
            yield cstring
        strings.append((cstring, cstate))
    i = 0
    while strings:
        cstring, cstate = strings.popleft()
        i += 1
        if cstate in self.map:
            for transition in sorted(self.map[cstate]):
                nstate = self.map[cstate][transition]
                if nstate in livestates:
                    for symbol in sorted(self.alphabet.by_transition[transition]):
                        nstring = cstring + [symbol]
                        if nstate in self.finals:
                            yield nstring
                        strings.append((nstring, nstate))
        if max_iterations is not None and i > max_iterations:
            raise ValueError(f"Couldn't find an example within {max_iterations} iterations")