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
def test_validators_validators(self):
    """
        As of v4.0.0, accessing jsonschema.validators.validators is
        deprecated.
        """
    message = 'Accessing jsonschema.validators.validators is deprecated'
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        value = validators.validators
    self.assertEqual(value, validators._VALIDATORS)
    self.assertEqual(w.filename, __file__)