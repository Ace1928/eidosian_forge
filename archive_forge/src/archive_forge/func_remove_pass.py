from functools import wraps
from inspect import unwrap
from typing import Callable, List, Optional
import logging
def remove_pass(self, _passes: List[Callable]):
    if _passes is None:
        return
    passes_left = []
    for ps in self.passes:
        if ps.__name__ not in _passes:
            passes_left.append(ps)
    self.passes = passes_left
    self._validated = False