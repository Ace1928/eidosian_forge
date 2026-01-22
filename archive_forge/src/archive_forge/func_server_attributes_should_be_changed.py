from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, scaleway_argument_spec, Scaleway
def server_attributes_should_be_changed(compute_api, target_server, wished_server):
    compute_api.module.debug('Checking if server attributes should be changed')
    compute_api.module.debug('Current Server: %s' % target_server)
    compute_api.module.debug('Wished Server: %s' % wished_server)
    debug_dict = dict(((x, (target_server[x], wished_server[x])) for x in PATCH_MUTABLE_SERVER_ATTRIBUTES if x in target_server and x in wished_server))
    compute_api.module.debug('Debug dict %s' % debug_dict)
    try:
        for key in PATCH_MUTABLE_SERVER_ATTRIBUTES:
            if key in target_server and key in wished_server:
                if isinstance(target_server[key], dict) and wished_server[key] and ('id' in target_server[key].keys()) and (target_server[key]['id'] != wished_server[key]):
                    return True
                elif not isinstance(target_server[key], dict) and target_server[key] != wished_server[key]:
                    return True
        return False
    except AttributeError:
        compute_api.module.fail_json(msg='Error while checking if attributes should be changed')