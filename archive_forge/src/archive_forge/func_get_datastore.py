from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def get_datastore(module, client, predicate):
    pool = client.datastorepool.info()
    found = 0
    found_datastore = None
    datastore_name = ''
    for datastore in pool.DATASTORE:
        if predicate(datastore):
            found = found + 1
            found_datastore = datastore
            datastore_name = datastore.NAME
    if found == 0:
        return None
    elif found > 1:
        module.fail_json(msg='There are more datastores with name: ' + datastore_name)
    return found_datastore