from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_parseEqualSignInParamValue(self):
    """
        Every C{=}-sign after the first in a query parameter is simply included
        in the value of the parameter.
        """
    u = URL.from_text('http://localhost/?=x=x=x')
    self.assertEqual(u.get(''), ['x=x=x'])
    self.assertEqual(u.to_text(), 'http://localhost/?=x=x=x')
    u = URL.from_text('http://localhost/?foo=x=x=x&bar=y')
    self.assertEqual(u.query, (('foo', 'x=x=x'), ('bar', 'y')))
    self.assertEqual(u.to_text(), 'http://localhost/?foo=x=x=x&bar=y')
    u = URL.from_text('https://example.com/?argument=3&argument=4&operator=%3D')
    iri = u.to_iri()
    self.assertEqual(iri.get('operator'), ['='])
    self.assertEqual(iri.to_uri().get('operator'), ['='])