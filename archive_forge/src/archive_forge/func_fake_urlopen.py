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
@contextmanager
def fake_urlopen(request):
    self.assertIsInstance(request, urllib.request.Request)
    self.assertEqual(request.full_url, 'http://bar')
    (header, value), = request.header_items()
    self.assertEqual(header.lower(), 'user-agent')
    self.assertEqual(value, 'python-jsonschema (deprecated $ref resolution)')
    yield BytesIO(json.dumps(schema).encode('utf8'))