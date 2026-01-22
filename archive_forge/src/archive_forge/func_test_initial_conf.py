from __future__ import (absolute_import, division, print_function)
import os
from contextlib import contextmanager
import pexpect
def test_initial_conf(tmpdir):
    conf = tmpdir.join('dr.conf')
    conf.write(INITIAL_CONF)
    with generator(tmpdir) as gen:
        gen.expect('override')
        gen.sendline('y')
        assert os.path.exists('/tmp/dr_ovirt-ansible/mapping_vars.yml')