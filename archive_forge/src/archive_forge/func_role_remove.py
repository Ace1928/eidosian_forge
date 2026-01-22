from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def role_remove(module, client, db_name, role):
    exists = role_find(client, role, db_name)
    if exists:
        if module.check_mode:
            module.exit_json(changed=True, role=role)
        db = client[db_name]
        db.command('dropRole', role)
    else:
        module.exit_json(changed=False, role=role)