from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_badUTF8AsIRI(self) -> None:
    """
        Bad UTF-8 in a path segment, query parameter, or fragment results in
        that portion of the URI remaining percent-encoded in the IRI.
        """
    urlWithBinary = 'http://xn--9ca.com/%00%FF/%C3%A9'
    uri = URL.fromText(urlWithBinary)
    iri = uri.asIRI()
    expectedIRI = 'http://é.com/%00%FF/é'
    actualIRI = iri.asText()
    self.assertEqual(actualIRI, expectedIRI, f'{actualIRI!r} != {expectedIRI!r}')