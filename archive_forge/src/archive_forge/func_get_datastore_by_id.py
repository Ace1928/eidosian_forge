from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def get_datastore_by_id(module, client, datastore_id):
    return get_datastore(module, client, lambda datastore: datastore.ID == datastore_id)