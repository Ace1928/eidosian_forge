from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def replicaset_status(client, module):
    """
    Return the replicaset status document from MongoDB
    # https://docs.mongodb.com/manual/reference/command/replSetGetStatus/
    """
    rs = client.admin.command('replSetGetStatus')
    return rs