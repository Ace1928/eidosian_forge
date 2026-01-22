import pytest
from winrm import Session
def test_target_as_full_url():
    s = Session('http://windows-host:1111/wsman', auth=('john.smith', 'secret'))
    assert s.url == 'http://windows-host:1111/wsman'