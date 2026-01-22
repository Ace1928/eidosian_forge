from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def set_sysview_igmp(self):
    """set igmp of sysview"""
    if not self.changed:
        return
    version = self.addr_family
    state = self.state
    if state == 'present':
        configxmlstr = CE_NC_MERGE_IGMP_SYSVIEW % version
    else:
        configxmlstr = CE_NC_DELETE_IGMP_SYSVIEW % version
    conf_str = build_config_xml(configxmlstr)
    recv_xml = set_nc_config(self.module, conf_str)
    self._checkresponse_(recv_xml, 'SET_SYSVIEW_IGMP')