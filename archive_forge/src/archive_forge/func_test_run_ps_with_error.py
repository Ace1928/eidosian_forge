import pytest
from winrm import Session
def test_run_ps_with_error(protocol_fake):
    s = Session('windows-host', auth=('john.smith', 'secret'))
    s.protocol = protocol_fake
    r = s.run_ps('Write-Error "Error"')
    assert r.status_code == 1
    assert b'Write-Error "Error"' in r.std_err
    assert len(r.std_out) == 0