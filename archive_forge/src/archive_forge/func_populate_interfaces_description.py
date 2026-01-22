from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def populate_interfaces_description(self, data):
    tables = ciscosmb_split_to_tables(data)
    interface_table = ciscosmb_parse_table(tables[0], False)
    portchanel_table = ciscosmb_parse_table(tables[1], False)
    interfaces = self._populate_interfaces_description_interface(interface_table)
    self.facts['interfaces'] = ciscosmb_merge_dicts(self.facts['interfaces'], interfaces)
    interfaces = self._populate_interfaces_description_portchanel(portchanel_table)
    self.facts['interfaces'] = ciscosmb_merge_dicts(self.facts['interfaces'], interfaces)