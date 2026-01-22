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
def test_import_cli(self):
    """
        As of v4.17.0, importing jsonschema.cli is deprecated.
        """
    message = 'The jsonschema CLI is deprecated and will be removed '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        import jsonschema.cli
        importlib.reload(jsonschema.cli)
    self.assertEqual(w.filename, importlib.__file__)