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
def test_license(self):
    output = subprocess.check_output([sys.executable, '-m', 'pip', 'show', 'jsonschema'], stderr=subprocess.STDOUT)
    self.assertIn(b'License: MIT', output)