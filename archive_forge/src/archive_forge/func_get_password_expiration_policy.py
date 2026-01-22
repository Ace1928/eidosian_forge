from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def get_password_expiration_policy(cursor, user, host, maria_role=False):
    """Function to get password policy for user.

    Args:
        cursor (cursor): DB driver cursor object.
        user (str): User name.
        host (str): User hostname.
        maria_role (bool, optional): mariadb or mysql. Defaults to False.

    Returns:
        policy (int): Current users password policy.
    """
    if not maria_role:
        statement = ('SELECT IFNULL(password_lifetime, -1) FROM mysql.user             WHERE User = %s AND Host = %s', (user, host))
    else:
        statement = ("SELECT JSON_EXTRACT(Priv, '$.password_lifetime') AS password_lifetime             FROM mysql.global_priv             WHERE User = %s AND Host = %s", (user, host))
    cursor.execute(*statement)
    policy = cursor.fetchone()[0]
    return int(policy)