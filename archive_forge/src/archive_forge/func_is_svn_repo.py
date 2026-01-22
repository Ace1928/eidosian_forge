from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.compat.version import LooseVersion
def is_svn_repo(self):
    """Checks if path is a SVN Repo."""
    rc = self._exec(['info', self.dest], check_rc=False)
    return rc == 0