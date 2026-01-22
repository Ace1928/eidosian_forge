from __future__ import absolute_import, division, print_function
import os
import warnings
from re import match
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.database import SQLParseError, mysql_quote_identifier
from ansible_collections.community.mysql.plugins.module_utils.mysql import mysql_connect, mysql_driver, mysql_driver_fail_msg, mysql_common_argument_spec
from ansible.module_utils._text import to_native
def getvariable(cursor, mysqlvar):
    cursor.execute('SHOW VARIABLES WHERE Variable_name = %s', (mysqlvar,))
    mysqlvar_val = cursor.fetchall()
    if len(mysqlvar_val) == 1:
        return mysqlvar_val[0][1]
    else:
        return None