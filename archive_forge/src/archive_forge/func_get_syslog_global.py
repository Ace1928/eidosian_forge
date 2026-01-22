from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config, check_ip_addr
def get_syslog_global(self):
    """get syslog global attributes"""
    cur_global_info = dict()
    conf_str = CE_NC_GET_CENTER_GLOBAL_INFO_HEADER
    if self.info_center_enable:
        conf_str += '<icEnable></icEnable>'
    if self.packet_priority:
        conf_str += '<packetPriority></packetPriority>'
    if self.suppress_enable:
        conf_str += '<suppressEnable></suppressEnable>'
    conf_str += CE_NC_GET_CENTER_GLOBAL_INFO_TAIL
    xml_str = get_nc_config(self.module, conf_str)
    if '<data/>' in xml_str:
        return cur_global_info
    else:
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        global_info = root.findall('syslog/globalParam')
        if global_info:
            for tmp in global_info:
                for site in tmp:
                    if site.tag in ['icEnable', 'packetPriority', 'suppressEnable']:
                        cur_global_info[site.tag] = site.text
        return cur_global_info