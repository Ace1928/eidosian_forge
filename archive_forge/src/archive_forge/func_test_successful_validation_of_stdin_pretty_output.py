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
def test_successful_validation_of_stdin_pretty_output(self):
    self.assertOutputs(files=dict(some_schema='{}'), stdin=StringIO('{}'), argv=['--output', 'pretty', 'some_schema'], stdout='===[SUCCESS]===(<stdin>)===\n', stderr='')