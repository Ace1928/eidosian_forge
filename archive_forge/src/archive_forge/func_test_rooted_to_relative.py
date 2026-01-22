from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_rooted_to_relative(self):
    """
        On host-relative URLs, the C{rooted} flag can be updated to indicate
        that the path should no longer be treated as absolute.
        """
    a = URL(path=['hello'])
    self.assertEqual(a.to_text(), 'hello')
    b = a.replace(rooted=True)
    self.assertEqual(b.to_text(), '/hello')
    self.assertNotEqual(a, b)