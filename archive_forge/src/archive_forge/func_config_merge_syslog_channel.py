from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config, check_ip_addr
def config_merge_syslog_channel(self, channel_id, channel_name):
    """config channel id"""
    if not self.is_exist_channel_id_name(channel_id, channel_name):
        conf_str = CE_NC_MERGE_CHANNEL_INFO_HEADER
        if channel_id:
            conf_str += '<icChnlId>%s</icChnlId>' % channel_id
        if channel_name:
            conf_str += '<icChnlCfgName>%s</icChnlCfgName>' % channel_name
        conf_str += CE_NC_MERGE_CHANNEL_INFO_TAIL
        recv_xml = set_nc_config(self.module, conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: Merge syslog channel id failed.')
        self.updates_cmd.append('info-center channel %s name %s' % (channel_id, channel_name))
        self.changed = True