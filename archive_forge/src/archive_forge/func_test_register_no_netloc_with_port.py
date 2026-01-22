from __future__ import unicode_literals
from typing import cast
from .. import _url
from .common import HyperlinkTestCase
from .._url import register_scheme, URL, DecodedURL
def test_register_no_netloc_with_port(self):
    with self.assertRaises(ValueError):
        register_scheme('badnetlocless', uses_netloc=False, default_port=7)