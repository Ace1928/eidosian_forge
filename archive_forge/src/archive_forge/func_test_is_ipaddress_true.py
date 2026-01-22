from test import notPyPy2
import mock
import pytest
from urllib3.exceptions import SNIMissingWarning
from urllib3.util import ssl_
@pytest.mark.parametrize('addr', ['::1', '::', 'FE80::8939:7684:D84b:a5A4%251', '127.0.0.1', '8.8.8.8', b'127.0.0.1', 'FE80::8939:7684:D84b:a5A4%251', b'FE80::8939:7684:D84b:a5A4%251', 'FE80::8939:7684:D84b:a5A4%19', b'FE80::8939:7684:D84b:a5A4%19'])
def test_is_ipaddress_true(addr):
    assert ssl_.is_ipaddress(addr)