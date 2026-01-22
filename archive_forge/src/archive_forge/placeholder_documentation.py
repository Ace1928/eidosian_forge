from collections import defaultdict
import hashlib
from typing import Any, Dict, Tuple
from ray.tune.search.sample import Categorical, Domain, Function
from ray.tune.search.variant_generator import assign_value
from ray.util.annotations import DeveloperAPI
Some functions take a resolved spec dict as input.

        Note: Function placeholders are independently sampled during
        resolution. Therefore their random states are not restored.
        