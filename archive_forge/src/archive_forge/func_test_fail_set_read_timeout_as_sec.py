import pytest
from winrm.protocol import Protocol
def test_fail_set_read_timeout_as_sec():
    with pytest.raises(ValueError) as exc:
        Protocol('endpoint', username='username', password='password', read_timeout_sec='30a', operation_timeout_sec='29')
    assert str(exc.value) == "failed to parse read_timeout_sec as int: invalid literal for int() with base 10: '30a'"