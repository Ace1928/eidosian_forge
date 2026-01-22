from __future__ import absolute_import, division, print_function
import re
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
def getfingerprint(self, keyfile):
    stdout, stderr = self.execute_command([self.gpg, '--no-tty', '--batch', '--with-colons', '--fixed-list-mode', '--with-fingerprint', keyfile])
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith('fpr:'):
            return line.split(':')[9]
    self.module.fail_json(msg='Unexpected gpg output')