from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.vexata import (
def get_eg(module, array):
    """Retrieve a named vg if it exists, None if absent."""
    name = module.params['name']
    try:
        egs = array.list_egs()
        eg = filter(lambda eg: eg['name'] == name, egs)
        if len(eg) == 1:
            return eg[0]
        else:
            return None
    except Exception:
        module.fail_json(msg='Error while attempting to retrieve export groups.')