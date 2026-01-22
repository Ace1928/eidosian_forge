from contextlib import redirect_stderr, redirect_stdout
from importlib import metadata
from io import StringIO
from json import JSONDecodeError
from pathlib import Path
from textwrap import dedent
from unittest import TestCase
import json
import os
import subprocess
import sys
import tempfile
import warnings
from jsonschema import Draft4Validator, Draft202012Validator
from jsonschema.exceptions import (
from jsonschema.validators import _LATEST_VERSION, validate
def test_custom_error_format_applies_to_schema_errors(self):
    instance, schema = (13, {'type': 12, 'minimum': 30})
    with self.assertRaises(SchemaError):
        validate(schema=schema, instance=instance)
    self.assertOutputs(files=dict(some_schema=json.dumps(schema)), argv=['--error-format', ':{error.message}._-_.{error.instance}:', 'some_schema'], exit_code=1, stderr=':12 is not valid under any of the given schemas._-_.12:')