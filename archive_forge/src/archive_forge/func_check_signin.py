from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import os
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
import platform
def check_signin(self):
    """ Verifies that the user is signed in to the Mac App Store """
    if self._checked_signin:
        return
    if LooseVersion(self._mac_version) >= LooseVersion(NOT_WORKING_MAC_VERSION_MAS_ACCOUNT):
        self.module.log('WARNING: You must be signed in via the Mac App Store GUI beforehand else error will occur')
    else:
        rc, out, err = self.run(['account'])
        if out.split('\n', 1)[0].rstrip() == 'Not signed in':
            self.module.fail_json(msg='You must be signed in to the Mac App Store')
    self._checked_signin = True