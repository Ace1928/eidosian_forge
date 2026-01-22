from __future__ import (absolute_import, division, print_function)
from functools import reduce
import os
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
from ansible_collections.community.mysql.plugins.module_utils.database import mysql_quote_identifier
def set_session_vars(module, cursor, session_vars):
    """Set session vars."""
    for var, value in session_vars.items():
        query = 'SET SESSION %s = ' % mysql_quote_identifier(var, 'vars')
        try:
            cursor.execute(query + '%s', (value,))
        except Exception as e:
            module.fail_json(msg='Failed to execute %s%s: %s' % (query, value, e))