import collections
import heapq
from itertools import chain
from typing import Deque, Dict, List, Optional, Set, Tuple
from .diff_tree import (
from .errors import MissingCommitError
from .objects import Commit, ObjectID, Tag
Possibly reorder a results iterator.

        Args:
          results: An iterator of WalkEntry objects, in the order returned
            from the queue_cls.
        Returns: An iterator or list of WalkEntry objects, in the order
            required by the Walker.
        