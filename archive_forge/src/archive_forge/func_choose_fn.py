import functools
import heapq
import math
import numbers
import time
from collections import deque
from . import helpers, paths
@property
def choose_fn(self):
    """The function that chooses which contraction to take - make this a
        property so that ``temperature`` and ``nbranch`` etc. can be updated
        between runs.
        """
    if self.nbranch == 1:
        return None
    return functools.partial(thermal_chooser, temperature=self.temperature, nbranch=self.nbranch, rel_temperature=self.rel_temperature)