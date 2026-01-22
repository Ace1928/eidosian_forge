from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def replicaset_good(statuses, module, votes):
    """
    Returns true if the replicaset is in a "good" condition.
    Good is defined as an odd number of servers >= 3, with
    max one primary, and any even amount of
    secondary and arbiter servers
    """
    msg = 'Unset'
    status = None
    valid_statuses = ['PRIMARY', 'SECONDARY', 'ARBITER']
    validate = module.params['validate']
    if validate == 'default':
        if len(statuses) % 2 == 1:
            if statuses.count('PRIMARY') == 1 and (statuses.count('SECONDARY') + statuses.count('ARBITER')) % 2 == 0 and (len(set(statuses) - set(valid_statuses)) == 0):
                status = True
                msg = 'replicaset is in a converged state'
            else:
                status = False
                msg = 'replicaset is not currently in a converged state'
        else:
            msg = 'Even number of servers in replicaset.'
            status = False
    elif validate == 'votes':
        if votes % 2 == 1:
            if statuses.count('PRIMARY') == 1 and len(set(statuses) - set(valid_statuses)) == 0:
                status = True
                msg = 'replicaset is in a converged state'
            else:
                status = False
                msg = 'replicaset is not currently in a converged state'
        else:
            msg = 'Even number of votes in replicaset.'
            status = False
    elif validate == 'minimal':
        if statuses.count('PRIMARY') == 1 and len(set(statuses) - set(valid_statuses)) == 0:
            status = True
            msg = 'replicaset is in a converged state'
        else:
            status = False
            msg = 'replicaset is not currently in a converged state'
    else:
        module.fail_json(msg='Invalid value for validate has been provided: {0}'.format(validate))
    return (status, msg)