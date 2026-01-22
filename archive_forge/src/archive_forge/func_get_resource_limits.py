from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def get_resource_limits(cursor, user, host):
    """Get user resource limits.

    Args:
        cursor (cursor): DB driver cursor object.
        user (str): User name.
        host (str): User host name.

    Returns: Dictionary containing current resource limits.
    """
    query = 'SELECT max_questions AS MAX_QUERIES_PER_HOUR, max_updates AS MAX_UPDATES_PER_HOUR, max_connections AS MAX_CONNECTIONS_PER_HOUR, max_user_connections AS MAX_USER_CONNECTIONS FROM mysql.user WHERE User = %s AND Host = %s'
    cursor.execute(query, (user, host))
    res = cursor.fetchone()
    if isinstance(res, dict):
        res = list(res.values())
    if not res:
        return None
    current_limits = {'MAX_QUERIES_PER_HOUR': res[0], 'MAX_UPDATES_PER_HOUR': res[1], 'MAX_CONNECTIONS_PER_HOUR': res[2], 'MAX_USER_CONNECTIONS': res[3]}
    cursor.execute('SELECT VERSION()')
    srv_type = cursor.fetchone()
    if isinstance(srv_type, dict):
        srv_type = list(srv_type.values())
    if 'mariadb' in srv_type[0].lower():
        query = 'SELECT max_statement_time AS MAX_STATEMENT_TIME FROM mysql.user WHERE User = %s AND Host = %s'
        cursor.execute(query, (user, host))
        res_max_statement_time = cursor.fetchone()
        if isinstance(res_max_statement_time, dict):
            res_max_statement_time = list(res_max_statement_time.values())
        current_limits['MAX_STATEMENT_TIME'] = res_max_statement_time[0]
    return current_limits