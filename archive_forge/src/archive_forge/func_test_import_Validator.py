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
def test_import_Validator(self):
    """
        As of v4.19.0, importing Validator from the package root is
        deprecated in favor of doing so from jsonschema.protocols.
        """
    message = 'Importing Validator directly from the jsonschema package '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        from jsonschema import Validator
    self.assertEqual(Validator, protocols.Validator)
    self.assertEqual(w.filename, __file__)