from __future__ import absolute_import, division, print_function
import os
import warnings
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible.module_utils._text import to_native
def stop_replica(module, cursor, connection_name='', channel='', fail_on_error=False, term='REPLICA'):
    if connection_name:
        query = "STOP %s '%s'" % (term, connection_name)
    else:
        query = 'STOP %s' % term
    if channel:
        query += " FOR CHANNEL '%s'" % channel
    try:
        executed_queries.append(query)
        cursor.execute(query)
        stopped = True
    except mysql_driver.Warning as e:
        stopped = False
    except Exception as e:
        if fail_on_error:
            module.fail_json(msg='STOP REPLICA failed: %s' % to_native(e))
        stopped = False
    return stopped