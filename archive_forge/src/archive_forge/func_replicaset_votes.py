from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def replicaset_votes(config_document):
    """
    Return the number of votes in the replicaset
    """
    votes = 0
    for member in config_document['config']['members']:
        votes += member['votes']
    return votes