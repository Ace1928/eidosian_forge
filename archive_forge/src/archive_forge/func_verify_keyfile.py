from __future__ import (absolute_import, division, print_function)
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
def verify_keyfile(self, keyfile, keyid):
    """Verify that keyfile matches the specified key ID"""
    if keyfile is None:
        self.module.fail_json(msg='expected a key, got none')
    elif keyid is None:
        self.module.fail_json(msg='expected a key ID, got none')
    rc, stdout, stderr = self.module.run_command([self.gpg, '--with-colons', '--with-fingerprint', '--batch', '--no-tty', '--show-keys', keyfile], check_rc=True)
    extracted_keyid = None
    for line in stdout.splitlines():
        if line.startswith('fpr:'):
            extracted_keyid = line.split(':')[9]
            break
    if extracted_keyid != keyid:
        self.module.fail_json(msg='key ID does not match. expected %s, got %s' % (keyid, extracted_keyid))