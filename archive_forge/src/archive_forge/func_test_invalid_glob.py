from netaddr import (
def test_invalid_glob():
    assert not valid_glob('1.1.1.a')
    assert not valid_glob('1.1.1.1/32')
    assert not valid_glob('1.1.1.a-b')
    assert not valid_glob('1.1.a-b.*')