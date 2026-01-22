from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_bgp_as(module, vrf_name):
    as_val = None
    get_path = '%s=%s/%s/global/config' % (network_instance_path, vrf_name, protocol_bgp_path)
    request = {'path': get_path, 'method': GET}
    try:
        response = edit_config(module, to_request(module, request))
    except ConnectionError as exc:
        module.fail_json(msg=str(exc), code=exc.code)
    resp = response[0][1]
    if 'openconfig-network-instance:config' in resp and 'as' in resp['openconfig-network-instance:config']:
        as_val = resp['openconfig-network-instance:config']['as']
    return as_val