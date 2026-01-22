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
def test_invalid_schema_with_invalid_instance_pretty_output(self):
    instance, schema = (13, {'type': 12, 'minimum': 30})
    with self.assertRaises(SchemaError) as e:
        validate(schema=schema, instance=instance)
    error = str(e.exception)
    self.assertOutputs(files=dict(some_schema=json.dumps(schema), some_instance=json.dumps(instance)), argv=['--output', 'pretty', '-i', 'some_instance', 'some_schema'], exit_code=1, stderr='===[SchemaError]===(some_schema)===\n\n' + str(error) + '\n-----------------------------\n')