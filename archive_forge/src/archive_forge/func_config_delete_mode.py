from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def config_delete_mode(self, nve_name, mode):
    """nve mode"""
    if mode == 'mode-l3':
        if not self.is_nve_mode_exist(nve_name, mode):
            return
        cfg_xml = CE_NC_MERGE_NVE_MODE % (nve_name, 'mode-l2')
        recv_xml = set_nc_config(self.module, cfg_xml)
        self.check_response(recv_xml, 'DELETE_MODE')
        self.updates_cmd.append('interface %s' % nve_name)
        self.updates_cmd.append('undo mode l3')
        self.changed = True
    else:
        self.module.fail_json(msg='Error: Can not configure undo mode l2.')