from __future__ import (absolute_import, division, print_function)
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
def recv_key(self, keyring, keyid, keyserver):
    """Receives key via keyserver"""
    cmd = [self.pacman_key, '--gpgdir', keyring, '--keyserver', keyserver, '--recv-keys', keyid]
    self.module.run_command(cmd, check_rc=True)
    self.lsign_key(keyring, keyid)