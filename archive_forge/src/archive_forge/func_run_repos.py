from __future__ import absolute_import, division, print_function
import os
from fnmatch import fnmatch
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
def run_repos(self, arguments):
    """
        Execute `subscription-manager repos` with arguments and manage common errors
        """
    rc, out, err = self.module.run_command([self.rhsm_bin, 'repos'] + arguments, **self.rhsm_kwargs)
    if rc == 0 and out == 'This system has no repositories available through subscriptions.\n':
        self.module.fail_json(msg='This system has no repositories available through subscriptions')
    elif rc == 1:
        self.module.fail_json(msg='subscription-manager failed with the following error: %s' % err)
    else:
        return (rc, out, err)