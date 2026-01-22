from __future__ import division
import contextlib
import json
import numbers
from jsonschema import _utils, _validators
from jsonschema.compat import (
from jsonschema.exceptions import ErrorTree  # Backwards compat  # noqa: F401
from jsonschema.exceptions import RefResolutionError, SchemaError, UnknownType
@classmethod
def check_schema(cls, schema):
    for error in cls(cls.META_SCHEMA).iter_errors(schema):
        raise SchemaError.create_from(error)