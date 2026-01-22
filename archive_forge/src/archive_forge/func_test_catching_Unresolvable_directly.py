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
def test_catching_Unresolvable_directly(self):
    """
        This behavior is the intended behavior (i.e. it's not deprecated), but
        given we do "tricksy" things in the iterim to wrap exceptions in a
        multiple inheritance subclass, we need to be extra sure it works and
        stays working.
        """
    validator = validators.Draft202012Validator({'$ref': 'urn:nothing'})
    with self.assertRaises(referencing.exceptions.Unresolvable) as e:
        validator.validate(12)
    expected = referencing.exceptions.Unresolvable(ref='urn:nothing')
    self.assertEqual((e.exception, str(e.exception)), (expected, 'Unresolvable: urn:nothing'))