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
def test_Validator_subclassing(self):
    """
        As of v4.12.0, subclassing a validator class produces an explicit
        deprecation warning.

        This was never intended to be public API (and some comments over the
        years in issues said so, but obviously that's not a great way to make
        sure it's followed).

        A future version will explicitly raise an error.
        """
    message = 'Subclassing validator classes is '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:

        class Subclass(validators.Draft202012Validator):
            pass
    self.assertEqual(w.filename, __file__)
    with self.assertWarnsRegex(DeprecationWarning, message) as w:

        class AnotherSubclass(validators.create(meta_schema={})):
            pass