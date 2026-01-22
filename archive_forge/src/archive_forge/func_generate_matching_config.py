import importlib
import os
import sys
from collections import namedtuple
from dataclasses import fields
from typing import Any, Callable, Dict, List
def generate_matching_config(superset: Dict[str, Any], config_class: Any) -> Any:
    """Given a superset of the inputs and a reference config class,
    return exactly the needed config"""
    field_names = list(map(lambda x: x.name, fields(config_class)))
    subset = {k: v for k, v in superset.items() if k in field_names}
    for k in field_names:
        if k not in subset.keys():
            subset[k] = None
    return config_class(**subset)