from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems, string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.l3_interfaces.l3_interfaces import (
def parse_l3_if_resources(self, l3_if_resources):
    l3_ifaces = []
    for iface in l3_if_resources:
        int_have = self._get_xml_dict(iface)
        int_dict = int_have['interface']
        if 'unit' in int_dict.keys() and int_dict.get('unit') is not None:
            unit_list = int_dict['unit']
            if isinstance(unit_list, list):
                for item in unit_list:
                    fact_dict = self._render_l3_intf(item, int_dict)
                    if fact_dict:
                        l3_ifaces.append(fact_dict)
            else:
                fact_dict = self._render_l3_intf(unit_list, int_dict)
                if fact_dict:
                    l3_ifaces.append(fact_dict)
    return l3_ifaces