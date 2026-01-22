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
def test_RefResolutionError(self):
    """
        As of v4.18.0, RefResolutionError is deprecated in favor of directly
        catching errors from the referencing library.
        """
    message = 'jsonschema.exceptions.RefResolutionError is deprecated'
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        from jsonschema import RefResolutionError
    self.assertEqual(RefResolutionError, exceptions._RefResolutionError)
    self.assertEqual(w.filename, __file__)
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        from jsonschema.exceptions import RefResolutionError
    self.assertEqual(RefResolutionError, exceptions._RefResolutionError)
    self.assertEqual(w.filename, __file__)