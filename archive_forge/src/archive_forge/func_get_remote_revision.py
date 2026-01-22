from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.compat.version import LooseVersion
def get_remote_revision(self):
    """Revision and URL of subversion working directory."""
    text = '\n'.join(self._exec(['info', self.repo]))
    rev = re.search(self.REVISION_RE, text, re.MULTILINE)
    if rev:
        rev = rev.group(0)
    else:
        rev = 'Unable to get remote revision'
    return rev