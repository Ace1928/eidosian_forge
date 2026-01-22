from __future__ import absolute_import, division, print_function
from uuid import UUID
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def get_roles_info(self, dbname):
    """Gather information about roles.

        Args:
            dbname (str): Database name to get role info from.

        Returns a dictionary with role information for the given db.
        """
    db = self.client[dbname]
    result = db.command({'rolesInfo': 1, 'showBuiltinRoles': True})['roles']
    roles_dict = {}
    for elem in result:
        roles_dict[elem['role']] = {}
        for key, val in iteritems(elem):
            if key in ['role', 'db']:
                continue
            roles_dict[elem['role']][key] = val
    return {dbname: roles_dict}