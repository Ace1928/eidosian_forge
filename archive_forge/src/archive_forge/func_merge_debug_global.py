from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def merge_debug_global(self):
    """ Merge debug global """
    conf_str = CE_MERGE_DEBUG_GLOBAL_HEADER
    if self.debug_time_stamp:
        conf_str += '<debugTimeStamp>%s</debugTimeStamp>' % self.debug_time_stamp.upper()
    conf_str += CE_MERGE_DEBUG_GLOBAL_TAIL
    recv_xml = set_nc_config(self.module, conf_str)
    if '<ok/>' not in recv_xml:
        self.module.fail_json(msg='Error: Merge debug global failed.')
    if self.debug_time_stamp:
        cmd = 'info-center timestamp debugging ' + TIME_STAMP_DICT.get(self.debug_time_stamp)
        self.updates_cmd.append(cmd)
    self.changed = True