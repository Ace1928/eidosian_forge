from __future__ import absolute_import, division, print_function
import os
import warnings
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible.module_utils._text import to_native
def get_replica_status(cursor, connection_name='', channel='', term='REPLICA'):
    if connection_name:
        query = "SHOW %s '%s' STATUS" % (term, connection_name)
    else:
        query = 'SHOW %s STATUS' % term
    if channel:
        query += " FOR CHANNEL '%s'" % channel
    cursor.execute(query)
    replica_status = cursor.fetchone()
    return replica_status