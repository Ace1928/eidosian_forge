import pytest
from winrm import Session
def test_target_as_hostname_then_port():
    s = Session('windows-host:1111', auth=('john.smith', 'secret'))
    assert s.url == 'http://windows-host:1111/wsman'