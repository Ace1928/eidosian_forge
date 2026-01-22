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
def test_invalid_schema_with_invalid_instance(self):
    """
        "Validating" an instance that's invalid under an invalid schema
        just shows the schema error.
        """
    self.assertOutputs(files=dict(some_schema='{"type": 12, "minimum": 30}', some_instance='13'), argv=['-i', 'some_instance', 'some_schema'], exit_code=1, stderr='                12: 12 is not valid under any of the given schemas\n            ')