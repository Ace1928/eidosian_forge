from __future__ import (absolute_import, division, print_function)
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.argspec.system.system import SystemArgs
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.facts.facts import FactsBase
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.facts.system.system import SystemFacts
def get_network_legacy_facts(self, fact_legacy_obj_map, legacy_facts_type=None):
    if not legacy_facts_type:
        legacy_facts_type = self._gather_subset
    runable_subsets = self.gen_runable(legacy_facts_type, frozenset(fact_legacy_obj_map.keys()))
    if runable_subsets:
        self.ansible_facts['ansible_net_gather_subset'] = []
        instances = list()
        for subset, valid_subset in runable_subsets:
            instances.append(fact_legacy_obj_map[valid_subset](self._module, self._fos, subset))
        for inst in instances:
            inst.populate_facts(self._connection, self.ansible_facts)