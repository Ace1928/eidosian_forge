from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def populate_addresses_ipv6(self, data):
    tables = ciscosmb_split_to_tables(data)
    ip_table = ciscosmb_parse_table(tables[0])
    ips = self._populate_address_ipv6(ip_table)
    self.facts['all_ipv6_addresses'] = ips