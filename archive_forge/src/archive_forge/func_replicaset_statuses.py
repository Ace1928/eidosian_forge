from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def replicaset_statuses(members_document, module):
    """
    Return a list of the statuses
    """
    statuses = []
    for member in members_document:
        statuses.append(members_document[member])
    return statuses