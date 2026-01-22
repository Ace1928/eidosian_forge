from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def member_status(client):
    """
    Return the member status string
    # https://docs.mongodb.com/manual/reference/command/replSetGetStatus/
    """
    myStateStr = None
    rs = client.admin.command('replSetGetStatus')
    for member in rs['members']:
        if 'self' in member.keys():
            myStateStr = member['stateStr']
    return myStateStr