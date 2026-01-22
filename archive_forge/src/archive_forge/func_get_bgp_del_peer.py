from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def get_bgp_del_peer(self, **kwargs):
    """ get_bgp_del_peer """
    module = kwargs['module']
    peerip = module.params['peer_addr']
    vrf_name = module.params['vrf_name']
    if vrf_name:
        if len(vrf_name) > 31 or len(vrf_name) == 0:
            module.fail_json(msg='Error: The len of vrf_name %s is out of [1 - 31].' % vrf_name)
    conf_str = CE_GET_BGP_PEER_HEADER % (vrf_name, peerip) + CE_GET_BGP_PEER_TAIL
    xml_str = self.netconf_get_config(module=module, conf_str=conf_str)
    result = list()
    if '<data/>' in xml_str:
        return result
    else:
        re_find = re.findall('.*<peerAddr>(.*)</peerAddr>.*', xml_str)
        if re_find:
            return re_find
        else:
            return result