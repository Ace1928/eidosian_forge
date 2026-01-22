from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
Check if a replicaset exists.

    Args:
        client (cursor): Mongodb cursor on admin database.
        cluster_cmd (str): Either isMaster or hello

    Returns:
        str: when the node is a member of a replicaset , False otherwise.
    