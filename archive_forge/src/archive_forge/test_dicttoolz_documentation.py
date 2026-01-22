from collections import defaultdict as _defaultdict
from collections.abc import Mapping
import os
from toolz.dicttoolz import (merge, merge_with, valmap, keymap, update_in,
from toolz.functoolz import identity
from toolz.utils import raises
Test CustomMapping as input and factory

    Class attributes:
        D: callable that inputs a dict and creates or returns a MutableMapping
        kw: kwargs dict to specify "factory" keyword (if applicable)
    