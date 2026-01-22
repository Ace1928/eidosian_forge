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
def test_validators_meta_schemas(self):
    """
        As of v4.0.0, accessing jsonschema.validators.meta_schemas is
        deprecated.
        """
    message = 'Accessing jsonschema.validators.meta_schemas is deprecated'
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        value = validators.meta_schemas
    self.assertEqual(value, validators._META_SCHEMAS)
    self.assertEqual(w.filename, __file__)