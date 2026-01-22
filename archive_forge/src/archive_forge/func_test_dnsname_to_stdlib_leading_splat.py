import os
import mock
import pytest
from ..test_util import TestUtilSSL  # noqa: E402, F401
from ..with_dummyserver.test_https import (  # noqa: E402, F401
from ..with_dummyserver.test_socketlevel import (  # noqa: E402, F401
def test_dnsname_to_stdlib_leading_splat(self):
    """
        If there's a wildcard character in the front of the string we handle it
        appropriately.
        """
    name = u'*.उदाहरण.परीक'
    expected_result = '*.xn--p1b6ci4b4b3a.xn--11b5bs8d'
    assert _dnsname_to_stdlib(name) == expected_result