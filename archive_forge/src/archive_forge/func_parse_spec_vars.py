import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
def parse_spec_vars(spec: Dict) -> Tuple[List[Tuple[Tuple, Any]], List[Tuple[Tuple, Any]], List[Tuple[Tuple, Any]]]:
    resolved, unresolved = _split_resolved_unresolved_values(spec)
    resolved_vars = list(resolved.items())
    if not unresolved:
        return (resolved_vars, [], [])
    grid_vars = []
    domain_vars = []
    for path, value in unresolved.items():
        if value.is_grid():
            grid_vars.append((path, value))
        else:
            domain_vars.append((path, value))
    grid_vars.sort()
    return (resolved_vars, domain_vars, grid_vars)