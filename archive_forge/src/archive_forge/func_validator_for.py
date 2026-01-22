from __future__ import division
import contextlib
import json
import numbers
from jsonschema import _utils, _validators
from jsonschema.compat import (
from jsonschema.exceptions import ErrorTree  # Backwards compat  # noqa: F401
from jsonschema.exceptions import RefResolutionError, SchemaError, UnknownType
def validator_for(schema, default=_unset):
    if default is _unset:
        default = Draft4Validator
    return meta_schemas.get(schema.get(u'$schema', u''), default)