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
def test_nonexistent_file_with_explicit_base_uri(self):
    schema = '{"$ref": "someNonexistentFile.json#definitions/num"}'
    instance = '1'
    with self.assertRaises(_RefResolutionError) as e:
        self.assertOutputs(files=dict(some_schema=schema, some_instance=instance), argv=['-i', 'some_instance', '--base-uri', Path.cwd().as_uri(), 'some_schema'])
    error = str(e.exception)
    self.assertIn(f"{os.sep}someNonexistentFile.json'", error)