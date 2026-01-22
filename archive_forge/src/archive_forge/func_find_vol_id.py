from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def find_vol_id(module, system, vol_name):
    """ Find the ID of this vol """
    vol_url = f'volumes?name={vol_name}&fields=id'
    vol = system.api.get(path=vol_url)
    result = vol.get_json()['result']
    if len(result) != 1:
        module.fail_json(f"Cannot find a volume with name '{vol_name}'")
    vol_id = result[0]['id']
    return vol_id