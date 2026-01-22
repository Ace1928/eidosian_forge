from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def get_datastore_by_name(module, client, datastore_name):
    return get_datastore(module, client, lambda datastore: datastore.NAME == datastore_name)