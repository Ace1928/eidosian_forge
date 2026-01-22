from collections import namedtuple
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, Any, Dict, Iterable, Set, FrozenSet, Optional
from interegular import InvalidSyntax, REFlags
from interegular.fsm import FSM, Alphabet, anything_else
from interegular.patterns import Pattern, Unsupported, parse_pattern
from interegular.utils import logger, soft_repr
def get_example_overlap(self, a: Any, b: Any, max_time: float=None) -> ExampleCollision:
    pa, pb = (self._patterns[a], self._patterns[b])
    needed_pre = max(pa.prefix_postfix[0], pb.prefix_postfix[0])
    needed_post = max(pa.prefix_postfix[1], pb.prefix_postfix[1])
    alphabet = pa.get_alphabet(REFlags(0)).union(pb.get_alphabet(REFlags(0)))[0]
    fa, fb = (pa.to_fsm(alphabet, (needed_pre, needed_post)), pb.to_fsm(alphabet, (needed_pre, needed_post)))
    intersection = fa.intersection(fb)
    if max_time is None:
        max_iterations = None
    else:
        max_iterations = int((max_time - 0.09) / (1.4e-06 * len(alphabet)))
    try:
        text = next(intersection.strings(max_iterations))
    except StopIteration:
        raise ValueError(f'No overlap between {a} and {b} exists')
    text = ''.join((c if c != anything_else else '?' for c in text))
    if needed_post > 0:
        return ExampleCollision(text[:needed_pre], text[needed_pre:-needed_post], text[-needed_post:])
    else:
        return ExampleCollision(text[:needed_pre], text[needed_pre:], '')