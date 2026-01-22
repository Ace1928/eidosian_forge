import pytest
from winrm import Session
def test_decode_clixml_no_clixml():
    s = Session('windows-host.example.com', auth=('john.smith', 'secret'))
    msg = b'stderr line'
    expected = b'stderr line'
    actual = s._clean_error_msg(msg)
    assert actual == expected