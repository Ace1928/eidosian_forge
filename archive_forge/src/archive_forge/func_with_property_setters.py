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
def with_property_setters(cls: TSchemaBase) -> TSchemaBase:
    """
    Decorator to add property setters to a Schema class.
    """
    schema = cls.resolve_references()
    for prop, propschema in schema.get('properties', {}).items():
        setattr(cls, prop, _PropertySetter(prop, propschema))
    return cls