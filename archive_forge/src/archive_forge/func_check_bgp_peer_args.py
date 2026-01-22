from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def check_bgp_peer_args(self, **kwargs):
    """ check_bgp_peer_args """
    module = kwargs['module']
    result = dict()
    need_cfg = False
    vrf_name = module.params['vrf_name']
    if vrf_name:
        if len(vrf_name) > 31 or len(vrf_name) == 0:
            module.fail_json(msg='Error: The len of vrf_name %s is out of [1 - 31].' % vrf_name)
    peer_addr = module.params['peer_addr']
    if peer_addr:
        if not check_ip_addr(ipaddr=peer_addr):
            module.fail_json(msg='Error: The peer_addr %s is invalid.' % peer_addr)
        need_cfg = True
    remote_as = module.params['remote_as']
    if remote_as:
        if len(remote_as) > 11 or len(remote_as) < 1:
            module.fail_json(msg='Error: The len of remote_as %s is out of [1 - 11].' % remote_as)
        need_cfg = True
    result['need_cfg'] = need_cfg
    return result