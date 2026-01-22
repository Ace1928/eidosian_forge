from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def merge_process(self):
    """merge ospf process"""
    xml_area = ''
    xml_str = ''
    self.updates_cmd.append('ospf %s' % self.process_id)
    xml_nh = ''
    if self.nexthop_addr and self.is_nexthop_change():
        xml_nh = CE_NC_XML_MERGE_NEXTHOP % (self.nexthop_addr, self.nexthop_weight)
        self.updates_cmd.append('nexthop %s weight %s' % (self.nexthop_addr, self.nexthop_weight))
    xml_lb = ''
    if self.max_load_balance and self.ospf_info.get('maxLoadBalancing') != self.max_load_balance:
        xml_lb = CE_NC_XML_SET_LB % self.max_load_balance
        self.updates_cmd.append('maximum load-balancing %s' % self.max_load_balance)
    xml_topo = ''
    if xml_lb or xml_nh:
        xml_topo = CE_NC_XML_BUILD_MERGE_TOPO % (xml_nh + xml_lb)
    if self.area:
        self.updates_cmd.append('area %s' % self.get_area_ip())
        xml_network = ''
        xml_auth = ''
        if self.addr and self.mask:
            if not self.is_network_exist():
                xml_network += CE_NC_XML_MERGE_NETWORKS % (self.addr, self.get_wildcard_mask())
                self.updates_cmd.append('network %s %s' % (self.addr, self.get_wildcard_mask()))
        if self.auth_mode:
            xml_auth += CE_NC_XML_SET_AUTH_MODE % self.auth_mode
            if self.auth_mode == 'none':
                self.updates_cmd.append('undo authentication-mode')
            else:
                self.updates_cmd.append('authentication-mode %s' % self.auth_mode)
            if self.auth_mode == 'simple' and self.auth_text_simple:
                xml_auth += CE_NC_XML_SET_AUTH_TEXT_SIMPLE % self.auth_text_simple
                self.updates_cmd.pop()
                self.updates_cmd.append('authentication-mode %s %s' % (self.auth_mode, self.auth_text_simple))
            if self.auth_mode in ['hmac-sha256', 'hmac-sha256', 'md5']:
                if self.auth_key_id and self.auth_text_md5:
                    xml_auth += CE_NC_XML_SET_AUTH_MD5 % (self.auth_key_id, self.auth_text_md5)
                    self.updates_cmd.pop()
                    self.updates_cmd.append('authentication-mode %s %s %s' % (self.auth_mode, self.auth_key_id, self.auth_text_md5))
        if xml_network or xml_auth or (not self.is_area_exist()):
            xml_area += CE_NC_XML_BUILD_MERGE_AREA % (self.get_area_ip(), xml_network + xml_auth)
        elif self.is_area_exist():
            self.updates_cmd.pop()
        else:
            pass
    if xml_area or xml_topo:
        xml_str = CE_NC_XML_BUILD_MERGE_PROCESS % (self.process_id, xml_topo + xml_area)
        recv_xml = set_nc_config(self.module, xml_str)
        self.check_response(recv_xml, 'MERGE_PROCESS')
        self.changed = True