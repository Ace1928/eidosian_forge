from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def reassign(self, old_owners, fail_on_role):
    """Implements REASSIGN OWNED BY command.

        If success, set self.changed as True.

        Arguments:
            old_owners (list): The ownership of all the objects within
                the current database, and of all shared objects (databases, tablespaces),
                owned by these roles will be reassigned to self.role.
            fail_on_role (bool): If True, fail when a role from old_owners does not exist.
                Otherwise just warn and continue.
        """
    roles = []
    for r in old_owners:
        if self.check_role_exists(r, fail_on_role):
            roles.append('"%s"' % r)
    if not roles:
        return False
    old_owners = ','.join(roles)
    query = ['REASSIGN OWNED BY']
    query.append(old_owners)
    query.append('TO "%s"' % self.role)
    query = ' '.join(query)
    self.changed = exec_sql(self, query, return_bool=True)