from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def check_global_args(self):
    """ Check global args """
    need_cfg = False
    find_flag = False
    self.cur_global_cfg['global_cfg'] = []
    if self.debug_time_stamp:
        conf_str = CE_GET_DEBUG_GLOBAL_HEADER
        conf_str += '<debugTimeStamp></debugTimeStamp>'
        conf_str += CE_GET_DEBUG_GLOBAL_TAIL
        xml_str = get_nc_config(self.module, conf_str)
        if '<data/>' in xml_str:
            find_flag = False
        else:
            xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_str)
            global_cfg = root.findall('syslog/globalParam')
            if global_cfg:
                for tmp in global_cfg:
                    tmp_dict = dict()
                    for site in tmp:
                        if site.tag in ['debugTimeStamp']:
                            tmp_dict[site.tag] = site.text
                    self.cur_global_cfg['global_cfg'].append(tmp_dict)
            if self.cur_global_cfg['global_cfg']:
                for tmp in self.cur_global_cfg['global_cfg']:
                    find_flag = True
                    if tmp.get('debugTimeStamp').lower() != self.debug_time_stamp:
                        find_flag = False
                    if find_flag:
                        break
            else:
                find_flag = False
        if self.state == 'present':
            need_cfg = bool(not find_flag)
        else:
            need_cfg = bool(find_flag)
    self.cur_global_cfg['need_cfg'] = need_cfg