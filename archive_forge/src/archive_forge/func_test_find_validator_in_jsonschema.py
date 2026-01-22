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
def test_find_validator_in_jsonschema(self):
    arguments = cli.parse_args(['--validator', 'Draft4Validator', '--instance', 'mem://some/instance', 'mem://some/schema'])
    self.assertIs(arguments['validator'], Draft4Validator)