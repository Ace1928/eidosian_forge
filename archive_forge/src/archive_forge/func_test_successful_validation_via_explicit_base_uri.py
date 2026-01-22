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
def test_successful_validation_via_explicit_base_uri(self):
    ref_schema_file = tempfile.NamedTemporaryFile(delete=False)
    ref_schema_file.close()
    self.addCleanup(os.remove, ref_schema_file.name)
    ref_path = Path(ref_schema_file.name)
    ref_path.write_text('{"definitions": {"num": {"type": "integer"}}}')
    schema = f'{{"$ref": "{ref_path.name}#/definitions/num"}}'
    self.assertOutputs(files=dict(some_schema=schema, some_instance='1'), argv=['-i', 'some_instance', '--base-uri', ref_path.parent.as_uri() + '/', 'some_schema'], stdout='', stderr='')