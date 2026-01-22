import os
import mock
import pytest
from ..test_util import TestUtilSSL  # noqa: E402, F401
from ..with_dummyserver.test_https import (  # noqa: E402, F401
from ..with_dummyserver.test_socketlevel import (  # noqa: E402, F401
def test_dnsname_to_stdlib_simple(self):
    """
        We can convert a dnsname to a native string when the domain is simple.
        """
    name = u'उदाहरण.परीक'
    expected_result = 'xn--p1b6ci4b4b3a.xn--11b5bs8d'
    assert _dnsname_to_stdlib(name) == expected_result