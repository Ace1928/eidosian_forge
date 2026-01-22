from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def check_source_args(self):
    """ Check source args """
    need_cfg = False
    find_flag = False
    self.cur_source_cfg['source_cfg'] = []
    if self.module_name:
        if len(self.module_name) < 1 or len(self.module_name) > 31:
            self.module.fail_json(msg='Error: The module_name is out of [1 - 31].')
        if not self.channel_id:
            self.module.fail_json(msg='Error: Please input channel_id at the same time.')
        if self.channel_id:
            if self.channel_id.isdigit():
                if int(self.channel_id) < 0 or int(self.channel_id) > 9:
                    self.module.fail_json(msg='Error: The value of channel_id is out of [0 - 9].')
            else:
                self.module.fail_json(msg='Error: The channel_id is not digit.')
        conf_str = CE_GET_DEBUG_SOURCE_HEADER
        if self.module_name != 'default':
            conf_str += '<moduleName>%s</moduleName>' % self.module_name.upper()
        else:
            conf_str += '<moduleName>default</moduleName>'
        if self.channel_id:
            conf_str += '<icChannelId></icChannelId>'
        if self.debug_enable != 'no_use':
            conf_str += '<dbgEnFlg></dbgEnFlg>'
        if self.debug_level:
            conf_str += '<dbgEnLevel></dbgEnLevel>'
        conf_str += CE_GET_DEBUG_SOURCE_TAIL
        xml_str = get_nc_config(self.module, conf_str)
        if '<data/>' in xml_str:
            find_flag = False
        else:
            xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_str)
            source_cfg = root.findall('syslog/icSources/icSource')
            if source_cfg:
                for tmp in source_cfg:
                    tmp_dict = dict()
                    for site in tmp:
                        if site.tag in ['moduleName', 'icChannelId', 'dbgEnFlg', 'dbgEnLevel']:
                            tmp_dict[site.tag] = site.text
                    self.cur_source_cfg['source_cfg'].append(tmp_dict)
            if self.cur_source_cfg['source_cfg']:
                for tmp in self.cur_source_cfg['source_cfg']:
                    find_flag = True
                    if self.module_name and tmp.get('moduleName').lower() != self.module_name.lower():
                        find_flag = False
                    if self.channel_id and tmp.get('icChannelId') != self.channel_id:
                        find_flag = False
                    if self.debug_enable != 'no_use' and tmp.get('dbgEnFlg') != self.debug_enable:
                        find_flag = False
                    if self.debug_level and tmp.get('dbgEnLevel') != self.debug_level:
                        find_flag = False
                    if find_flag:
                        break
            else:
                find_flag = False
        if self.state == 'present':
            need_cfg = bool(not find_flag)
        else:
            need_cfg = bool(find_flag)
    self.cur_source_cfg['need_cfg'] = need_cfg