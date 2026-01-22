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
def run_cli(self, argv, files=None, stdin=StringIO(), exit_code=0, **override):
    arguments = cli.parse_args(argv)
    arguments.update(override)
    self.assertFalse(hasattr(cli, 'open'))
    cli.open = fake_open(files or {})
    try:
        stdout, stderr = (StringIO(), StringIO())
        actual_exit_code = cli.run(arguments, stdin=stdin, stdout=stdout, stderr=stderr)
    finally:
        del cli.open
    self.assertEqual(actual_exit_code, exit_code, msg=dedent(f'\n                    Expected an exit code of {exit_code} != {actual_exit_code}.\n\n                    stdout: {stdout.getvalue()}\n\n                    stderr: {stderr.getvalue()}\n                '))
    return (stdout.getvalue(), stderr.getvalue())