from __future__ import absolute_import, division, print_function
import os
import warnings
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible.module_utils._text import to_native
def get_primary_status(cursor):
    cursor.execute('SHOW MASTER STATUS')
    primarystatus = cursor.fetchone()
    return primarystatus