from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def revoke_access_token(self):
    if self._module.check_mode:
        return True
    changed = False
    try:
        self.access_token_object.delete()
        changed = True
    except gitlab.exceptions.GitlabCreateError as e:
        self._module.fail_json(msg='Failed to revoke access token: %s ' % to_native(e))
    return changed