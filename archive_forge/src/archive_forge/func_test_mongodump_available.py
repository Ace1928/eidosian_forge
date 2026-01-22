import os
import testinfra.utils.ansible_runner
def test_mongodump_available(host):
    cmd = host.run('mongodump --version')
    assert cmd.rc == 0
    assert 'mongodump version' in cmd.stdout