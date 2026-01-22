from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
def set_default_role_all(self, user):
    """Run 'SET DEFAULT ROLE ALL TO' a user.

        The command is not supported by MariaDB, ignored.

        Args:
            user (tuple): User / role to run the command against in the form (username, hostname).
        """
    pass