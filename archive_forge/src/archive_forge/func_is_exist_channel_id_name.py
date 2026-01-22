from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config, check_ip_addr
def is_exist_channel_id_name(self, channel_id, channel_name):
    """if channel id exist"""
    if not self.channel_info:
        return False
    for id2name in self.channel_info['channelInfos']:
        if id2name['icChnlId'] == channel_id and id2name['icChnlCfgName'] == channel_name:
            return True
    return False