import collections
import contextlib
import copy
import inspect
import json
import sys
import textwrap
from typing import (
from itertools import zip_longest
from importlib.metadata import version as importlib_version
from typing import Final
import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import numpy as np
import pandas as pd
from packaging.version import Version
from altair import vegalite
@classmethod
def hash_schema(cls, schema: dict, use_json: bool=True) -> int:
    """
        Compute a python hash for a nested dictionary which
        properly handles dicts, lists, sets, and tuples.

        At the top level, the function excludes from the hashed schema all keys
        listed in `exclude_keys`.

        This implements two methods: one based on conversion to JSON, and one based
        on recursive conversions of unhashable to hashable types; the former seems
        to be slightly faster in several benchmarks.
        """
    if cls._hash_exclude_keys and isinstance(schema, dict):
        schema = {key: val for key, val in schema.items() if key not in cls._hash_exclude_keys}
    if use_json:
        s = json.dumps(schema, sort_keys=True)
        return hash(s)
    else:

        def _freeze(val):
            if isinstance(val, dict):
                return frozenset(((k, _freeze(v)) for k, v in val.items()))
            elif isinstance(val, set):
                return frozenset(map(_freeze, val))
            elif isinstance(val, list) or isinstance(val, tuple):
                return tuple(map(_freeze, val))
            else:
                return val
        return hash(_freeze(schema))