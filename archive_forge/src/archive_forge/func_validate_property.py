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
def validate_property(cls, name: str, value: Any, schema: Optional[dict]=None) -> None:
    """
        Validate a property against property schema in the context of the
        rootschema
        """
    value = _todict(value, context={})
    props = cls.resolve_references(schema or cls._schema).get('properties', {})
    return validate_jsonschema(value, props.get(name, {}), rootschema=cls._rootschema or cls._schema)