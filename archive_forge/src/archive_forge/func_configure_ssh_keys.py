from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
import time
from ansible.module_utils.basic import AnsibleModule
def configure_ssh_keys(self):
    rsa_key_file = '%s/root/etc/ssh/ssh_host_rsa_key' % self.path
    dsa_key_file = '%s/root/etc/ssh/ssh_host_dsa_key' % self.path
    if not os.path.isfile(rsa_key_file):
        cmd = '%s -f %s -t rsa -N ""' % (self.ssh_keygen_cmd, rsa_key_file)
        rc, out, err = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Failed to create rsa key. %s' % (out + err))
    if not os.path.isfile(dsa_key_file):
        cmd = '%s -f %s -t dsa -N ""' % (self.ssh_keygen_cmd, dsa_key_file)
        rc, out, err = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Failed to create dsa key. %s' % (out + err))