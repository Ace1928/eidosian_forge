from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def merge_debug_source(self):
    """ Merge debug source """
    conf_str = CE_MERGE_DEBUG_SOURCE_HEADER
    if self.module_name:
        conf_str += '<moduleName>%s</moduleName>' % self.module_name
    if self.channel_id:
        conf_str += '<icChannelId>%s</icChannelId>' % self.channel_id
    if self.debug_enable != 'no_use':
        conf_str += '<dbgEnFlg>%s</dbgEnFlg>' % self.debug_enable
    if self.debug_level:
        conf_str += '<dbgEnLevel>%s</dbgEnLevel>' % self.debug_level
    conf_str += CE_MERGE_DEBUG_SOURCE_TAIL
    recv_xml = set_nc_config(self.module, conf_str)
    if '<ok/>' not in recv_xml:
        self.module.fail_json(msg='Error: Merge debug source failed.')
    cmd = 'info-center source'
    if self.module_name:
        cmd += ' %s' % self.module_name
    if self.channel_id:
        cmd += ' channel %s' % self.channel_id
    if self.debug_enable != 'no_use':
        if self.debug_enable == 'true':
            cmd += ' debug state on'
        else:
            cmd += ' debug state off'
    if self.debug_level:
        cmd += ' level %s' % self.debug_level
    self.updates_cmd.append(cmd)
    self.changed = True