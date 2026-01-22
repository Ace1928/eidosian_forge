from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def config_merge_vni2bd(self, bd_id, vni_id):
    """config vni to bd id"""
    if self.is_vni_bd_change(vni_id, bd_id):
        cfg_xml = CE_NC_MERGE_VNI_BD_ID % (vni_id, bd_id)
        recv_xml = set_nc_config(self.module, cfg_xml)
        self.check_response(recv_xml, 'MERGE_VNI_BD')
        self.updates_cmd.append('bridge-domain %s' % bd_id)
        self.updates_cmd.append('vxlan vni %s' % vni_id)
        self.changed = True