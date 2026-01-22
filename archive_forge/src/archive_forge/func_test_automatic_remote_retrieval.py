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
def test_automatic_remote_retrieval(self):
    """
        Automatic retrieval of remote references is deprecated as of v4.18.0.
        """
    ref = 'http://bar#/$defs/baz'
    schema = {'$defs': {'baz': {'type': 'integer'}}}
    if 'requests' in sys.modules:
        self.addCleanup(sys.modules.__setitem__, 'requests', sys.modules['requests'])
    sys.modules['requests'] = None

    @contextmanager
    def fake_urlopen(request):
        self.assertIsInstance(request, urllib.request.Request)
        self.assertEqual(request.full_url, 'http://bar')
        (header, value), = request.header_items()
        self.assertEqual(header.lower(), 'user-agent')
        self.assertEqual(value, 'python-jsonschema (deprecated $ref resolution)')
        yield BytesIO(json.dumps(schema).encode('utf8'))
    validator = validators.Draft202012Validator({'$ref': ref})
    message = 'Automatically retrieving remote references '
    patch = mock.patch.object(urllib.request, 'urlopen', new=fake_urlopen)
    with patch, self.assertWarnsRegex(DeprecationWarning, message):
        self.assertEqual((validator.is_valid({}), validator.is_valid(37)), (False, True))