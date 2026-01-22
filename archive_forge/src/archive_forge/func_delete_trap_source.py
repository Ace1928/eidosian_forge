from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_trap_source(self):
    """ Delete trap source """
    if self.trap_enable == 'no_use' and (not self.trap_level):
        conf_str = CE_DELETE_TRAP_SOURCE_HEADER
        if self.module_name:
            conf_str += '<moduleName>%s</moduleName>' % self.module_name
        if self.channel_id:
            conf_str += '<icChannelId>%s</icChannelId>' % self.channel_id
        conf_str += CE_DELETE_TRAP_SOURCE_TAIL
    else:
        conf_str = CE_MERGE_TRAP_SOURCE_HEADER
        if self.module_name:
            conf_str += '<moduleName>%s</moduleName>' % self.module_name
        if self.channel_id:
            conf_str += '<icChannelId>%s</icChannelId>' % self.channel_id
        if self.trap_enable != 'no_use':
            conf_str += '<trapEnFlg>%s</trapEnFlg>' % CHANNEL_DEFAULT_TRAP_STATE.get(self.channel_id)
        if self.trap_level:
            conf_str += '<trapEnLevel>%s</trapEnLevel>' % CHANNEL_DEFAULT_TRAP_LEVEL.get(self.channel_id)
        conf_str += CE_MERGE_TRAP_SOURCE_TAIL
    recv_xml = self.netconf_set_config(conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        self.module.fail_json(msg='Error: Delete trap source failed.')
    cmd = 'undo info-center source'
    if self.module_name:
        cmd += ' %s' % self.module_name
    if self.channel_id:
        cmd += ' channel %s' % self.channel_id
    if self.trap_enable != 'no_use':
        cmd += ' trap state'
    if self.trap_level:
        cmd += ' level'
    self.updates_cmd.append(cmd)
    self.changed = True