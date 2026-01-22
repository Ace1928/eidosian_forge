import pytest
from jeepney import auth
def test_make_auth_external():
    b = auth.make_auth_external()
    assert b.startswith(b'AUTH EXTERNAL')