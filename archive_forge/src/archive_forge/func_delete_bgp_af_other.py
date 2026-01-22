from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def delete_bgp_af_other(self, **kwargs):
    """ delete_bgp_af_other """
    module = kwargs['module']
    vrf_name = module.params['vrf_name']
    af_type = module.params['af_type']
    conf_str = CE_MERGE_BGP_ADDRESS_FAMILY_HEADER % (vrf_name, af_type)
    cmds = []
    router_id = module.params['router_id']
    if router_id:
        conf_str += '<routerId></routerId>'
        cmd = 'undo router-id %s' % router_id
        cmds.append(cmd)
    determin_med = module.params['determin_med']
    if determin_med != 'no_use':
        conf_str += '<determinMed></determinMed>'
        cmd = 'undo deterministic-med'
        cmds.append(cmd)
    ebgp_if_sensitive = module.params['ebgp_if_sensitive']
    if ebgp_if_sensitive != 'no_use':
        conf_str += '<ebgpIfSensitive></ebgpIfSensitive>'
        cmd = 'undo ebgp-interface-sensitive'
        cmds.append(cmd)
    relay_delay_enable = module.params['relay_delay_enable']
    if relay_delay_enable != 'no_use':
        conf_str += '<relayDelayEnable></relayDelayEnable>'
    conf_str += CE_MERGE_BGP_ADDRESS_FAMILY_TAIL
    recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        module.fail_json(msg='Error: Merge bgp address family other agrus failed.')
    return cmds