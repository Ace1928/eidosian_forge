import sys
import pytest
from cheroot.cli import (
@pytest.mark.parametrize(('raw_bind_addr', 'expected_bind_addr'), (('192.168.1.1:80', ('192.168.1.1', 80)), ('[::1]:8000', ('::1', 8000)), ('localhost:5000', ('localhost', 5000)), ('foo@bar:5000', ('bar', 5000)), ('foo', ('foo', None)), ('123456789', ('123456789', None)), ('/tmp/cheroot.sock', '/tmp/cheroot.sock'), ('/tmp/some-random-file-name', '/tmp/some-random-file-name'), ('@cheroot', '\x00cheroot')))
def test_parse_wsgi_bind_addr(raw_bind_addr, expected_bind_addr):
    """Check the parsing of the --bind option.

    Verify some of the supported addresses and the expected return value.
    """
    assert parse_wsgi_bind_addr(raw_bind_addr) == expected_bind_addr