from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.compat.version import LooseVersion
def has_option_password_from_stdin(self):
    rc, version, err = self.module.run_command([self.svn_path, '--version', '--quiet'], check_rc=True)
    return LooseVersion(version) >= LooseVersion('1.10.0')