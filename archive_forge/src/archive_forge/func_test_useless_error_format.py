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
def test_useless_error_format(self):
    stdout, stderr = self.cli_output_for('--output', 'pretty', '--error-format', 'foo', 'mem://some/schema')
    self.assertIn('--error-format can only be used with --output plain', stderr)
    self.assertFalse(stdout)