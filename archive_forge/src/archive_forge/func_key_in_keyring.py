from __future__ import (absolute_import, division, print_function)
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
def key_in_keyring(self, keyring, keyid):
    """Check if the key ID is in pacman's keyring"""
    rc, stdout, stderr = self.module.run_command([self.gpg, '--with-colons', '--batch', '--no-tty', '--no-default-keyring', '--keyring=%s/pubring.gpg' % keyring, '--list-keys', keyid], check_rc=False)
    if rc != 0:
        if stderr.find('No public key') >= 0:
            return False
        else:
            self.module.fail_json(msg='gpg returned an error: %s' % stderr)
    return True