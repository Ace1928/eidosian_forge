from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def config_merge_vni_peer_ip(self, nve_name, vni_id, peer_ip_list):
    """config vni peer ip"""
    if self.is_vni_peer_list_change(nve_name, vni_id, peer_ip_list):
        cfg_xml = CE_NC_MERGE_VNI_PEER_ADDRESS_IP_HEAD % (nve_name, vni_id)
        for peer_ip in peer_ip_list:
            cfg_xml += CE_NC_MERGE_VNI_PEER_ADDRESS_IP_MERGE % peer_ip
        cfg_xml += CE_NC_MERGE_VNI_PEER_ADDRESS_IP_END
        recv_xml = set_nc_config(self.module, cfg_xml)
        self.check_response(recv_xml, 'MERGE_VNI_PEER_IP')
        self.updates_cmd.append('interface %s' % nve_name)
        for peer_ip in peer_ip_list:
            cmd_output = 'vni %s head-end peer-list %s' % (vni_id, peer_ip)
            self.updates_cmd.append(cmd_output)
        self.changed = True