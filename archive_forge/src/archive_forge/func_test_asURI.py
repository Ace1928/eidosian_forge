from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_asURI(self) -> None:
    """
        L{URL.asURI} produces an URI which converts any URI unicode encoding
        into pure US-ASCII and returns a new L{URL}.
        """
    unicodey = 'http://é.com/é?á=í#ú'
    iri = URL.fromText(unicodey)
    uri = iri.asURI()
    self.assertEqual(iri.host, 'é.com')
    self.assertEqual(iri.path[0], 'é')
    self.assertEqual(iri.asText(), unicodey)
    expectedURI = 'http://xn--9ca.com/%C3%A9?%C3%A1=%C3%AD#%C3%BA'
    actualURI = uri.asText()
    self.assertEqual(actualURI, expectedURI, f'{actualURI!r} != {expectedURI!r}')