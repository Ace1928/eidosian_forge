from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def delete_vlan(module, rest_obj, vlan_id):
    if module.check_mode:
        module.exit_json(changed=True, msg=CHECK_MODE_MSG)
    rest_obj.invoke_request('DELETE', VLAN_ID_CONFIG.format(Id=vlan_id))
    module.exit_json(msg='Successfully deleted the VLAN.', changed=True)