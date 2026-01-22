import pytest
from jeepney import auth
def test_make_auth_anonymous():
    b = auth.make_auth_anonymous()
    assert b.startswith(b'AUTH ANONYMOUS')