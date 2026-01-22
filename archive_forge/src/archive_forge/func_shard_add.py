from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def shard_add(client, shard):
    try:
        sh = client['admin'].command('addShard', shard)
    except Exception as excep:
        raise excep
    return sh