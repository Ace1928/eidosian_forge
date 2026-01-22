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
def cli_output_for(self, *argv):
    stdout, stderr = (StringIO(), StringIO())
    with redirect_stdout(stdout), redirect_stderr(stderr):
        with self.assertRaises(SystemExit):
            cli.parse_args(argv)
    return (stdout.getvalue(), stderr.getvalue())