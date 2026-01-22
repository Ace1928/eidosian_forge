from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
def different(self, other):
    """
            Two FSMs are considered different if they have a non-empty symmetric
            difference.
        """
    return not (self ^ other).empty()