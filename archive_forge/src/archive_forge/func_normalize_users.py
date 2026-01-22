from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
def normalize_users(module, users, is_mariadb=False):
    """Normalize passed user names.

    Example of transformation:
    ['user0'] => [('user0', '')] / ['user0'] => [('user0', '%')]
    ['user0@host0'] => [('user0', 'host0')]

    Args:
        module (AnsibleModule): Object of the AnsibleModule class.
        users (list): List of user names.
        is_mariadb (bool): Flag indicating we are working with MariaDB

    Returns:
        list: List of tuples like [('user0', ''), ('user0', 'host0')].
    """
    normalized_users = []
    for user in users:
        try:
            tmp = user.split('@')
            if tmp[0] == '':
                module.fail_json(msg="Member's name cannot be empty.")
            if len(tmp) == 1:
                if not is_mariadb:
                    normalized_users.append((tmp[0], '%'))
                else:
                    normalized_users.append((tmp[0], ''))
            elif len(tmp) == 2:
                normalized_users.append((tmp[0], tmp[1]))
        except Exception as e:
            msg = 'Error occured while parsing the name "%s": %s. It must be in the format "username" or "username@hostname" ' % (user, to_native(e))
            module.fail_json(msg=msg)
    return normalized_users