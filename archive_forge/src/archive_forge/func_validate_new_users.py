from __future__ import absolute_import, division, print_function
import json
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def validate_new_users(self, want, have):
    new_users = self.get_new_users(want, have)
    invalid_users = []
    for user in new_users:
        params = []
        if not user['role']:
            params.append('role')
        if not user['password']:
            params.append('password')
        if params:
            invalid_users.append({user['name']: params})
    if invalid_users:
        err_msg = 'Missing parameter(s) for new users! ' + str(invalid_users)
        self._module.fail_json(msg=err_msg, code=513)