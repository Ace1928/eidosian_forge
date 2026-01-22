import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
def increment(i):
    value_indices[i] += 1
    if value_indices[i] >= len(grid_vars[i][1]):
        value_indices[i] = 0
        if i + 1 < len(value_indices):
            return increment(i + 1)
        else:
            return True
    return False