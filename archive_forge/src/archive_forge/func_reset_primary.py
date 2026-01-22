from __future__ import absolute_import, division, print_function
import os
import warnings
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible.module_utils._text import to_native
def reset_primary(module, cursor, fail_on_error=False):
    query = 'RESET MASTER'
    try:
        executed_queries.append(query)
        cursor.execute(query)
        reset = True
    except mysql_driver.Warning as e:
        reset = False
    except Exception as e:
        if fail_on_error:
            module.fail_json(msg='RESET MASTER failed: %s' % to_native(e))
        reset = False
    return reset