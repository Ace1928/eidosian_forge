from contextlib import contextmanager
from io import BytesIO
from unittest import TestCase, mock
import importlib.metadata
import json
import subprocess
import sys
import urllib.request
import referencing.exceptions
from jsonschema import FormatChecker, exceptions, protocols, validators
def test_Validator_iter_errors_two_arguments(self):
    """
        As of v4.0.0, calling iter_errors with two arguments (to provide a
        different schema) is deprecated.
        """
    validator = validators.Draft7Validator({})
    message = 'Passing a schema to Validator.iter_errors is deprecated '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        error, = validator.iter_errors('foo', {'type': 'number'})
    self.assertEqual(error.validator, 'type')
    self.assertEqual(w.filename, __file__)