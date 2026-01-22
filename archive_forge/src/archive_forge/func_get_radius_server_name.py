from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def get_radius_server_name(self, **kwargs):
    """ Get radius server name """
    module = kwargs['module']
    radius_group_name = module.params['radius_group_name']
    radius_server_type = module.params['radius_server_type']
    radius_server_name = module.params['radius_server_name']
    radius_server_port = module.params['radius_server_port']
    radius_server_mode = module.params['radius_server_mode']
    radius_vpn_name = module.params['radius_vpn_name']
    state = module.params['state']
    result = dict()
    result['radius_server_name_cfg'] = []
    need_cfg = False
    conf_str = CE_GET_RADIUS_SERVER_NAME % radius_group_name
    recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
    if '<data/>' in recv_xml:
        if state == 'present':
            need_cfg = True
    else:
        xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        radius_server_name_cfg = root.findall('radius/rdsTemplates/rdsTemplate/rdsServerNames/rdsServerName')
        if radius_server_name_cfg:
            for tmp in radius_server_name_cfg:
                tmp_dict = dict()
                for site in tmp:
                    if site.tag in ['serverType', 'serverName', 'serverPort', 'serverMode', 'vpnName']:
                        tmp_dict[site.tag] = site.text
                result['radius_server_name_cfg'].append(tmp_dict)
        if result['radius_server_name_cfg']:
            cfg = dict()
            config_list = list()
            if radius_server_type:
                cfg['serverType'] = radius_server_type.lower()
            if radius_server_name:
                cfg['serverName'] = radius_server_name.lower()
            if radius_server_port:
                cfg['serverPort'] = radius_server_port.lower()
            if radius_server_mode:
                cfg['serverMode'] = radius_server_mode.lower()
            if radius_vpn_name:
                cfg['vpnName'] = radius_vpn_name.lower()
            for tmp in result['radius_server_name_cfg']:
                exist_cfg = dict()
                if radius_server_type:
                    exist_cfg['serverType'] = tmp.get('serverType').lower()
                if radius_server_name:
                    exist_cfg['serverName'] = tmp.get('serverName').lower()
                if radius_server_port:
                    exist_cfg['serverPort'] = tmp.get('serverPort').lower()
                if radius_server_mode:
                    exist_cfg['serverMode'] = tmp.get('serverMode').lower()
                if radius_vpn_name:
                    exist_cfg['vpnName'] = tmp.get('vpnName').lower()
                config_list.append(exist_cfg)
            if cfg in config_list:
                if state == 'present':
                    need_cfg = False
                else:
                    need_cfg = True
            elif state == 'present':
                need_cfg = True
            else:
                need_cfg = False
    result['need_cfg'] = need_cfg
    return result