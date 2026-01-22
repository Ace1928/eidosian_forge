from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_trap_global(self):
    """ Delete trap global """
    conf_str = CE_MERGE_TRAP_GLOBAL_HEADER
    if self.trap_time_stamp:
        conf_str += '<trapTimeStamp>DATE_SECOND</trapTimeStamp>'
    if self.trap_buff_enable != 'no_use':
        conf_str += '<icTrapBuffEn>false</icTrapBuffEn>'
    if self.trap_buff_size:
        conf_str += '<trapBuffSize>256</trapBuffSize>'
    conf_str += CE_MERGE_TRAP_GLOBAL_TAIL
    recv_xml = self.netconf_set_config(conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        self.module.fail_json(msg='Error: delete trap global failed.')
    if self.trap_time_stamp:
        cmd = 'undo info-center timestamp trap'
        self.updates_cmd.append(cmd)
    if self.trap_buff_enable != 'no_use':
        cmd = 'undo info-center trapbuffer'
        self.updates_cmd.append(cmd)
    if self.trap_buff_size:
        cmd = 'undo info-center trapbuffer size'
        self.updates_cmd.append(cmd)
    self.changed = True