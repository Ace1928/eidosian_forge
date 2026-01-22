import contextlib
import itertools
import re
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple
from . import branch as _mod_branch
from . import errors
from .inter import InterObject
from .registry import Registry
from .revision import RevisionID
def sort_alpha(branch, tags):
    """Sort tags lexicographically, in place.

    :param branch: Branch
    :param tags: List of tuples with tag name and revision id.
    """
    tags.sort()