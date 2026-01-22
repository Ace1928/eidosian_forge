from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.l2_interfaces import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import (
def process_list_attrs(self, param):
    if param:
        for _k, val in iteritems(param):
            val['name'] = normalize_interface(val['name'])
            if val.get('trunk'):
                for vlan in ['allowed_vlans', 'pruning_vlans']:
                    if val.get('trunk').get(vlan):
                        val['trunk'][vlan] = vlan_range_to_list(val.get('trunk').get(vlan))