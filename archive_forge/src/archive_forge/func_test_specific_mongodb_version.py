import os
import testinfra.utils.ansible_runner
def test_specific_mongodb_version(host):
    cmd = host.run('mongod --version')
    assert cmd.rc == 0
    assert '6.0.3' in cmd.stdout