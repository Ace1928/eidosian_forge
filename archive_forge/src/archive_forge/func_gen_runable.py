from __future__ import (absolute_import, division, print_function)
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.argspec.system.system import SystemArgs
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.facts.facts import FactsBase
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.facts.system.system import SystemFacts
def gen_runable(self, subsets, valid_subsets):
    """ Generate the runable subset

        :param module: The module instance
        :param subsets: The provided subsets
        :param valid_subsets: The valid subsets
        :rtype: list
        :returns: The runable subsets
        """
    runable_subsets = []
    FACT_DETAIL_SUBSETS = []
    FACT_DETAIL_SUBSETS.extend(SystemArgs.FACT_SYSTEM_SUBSETS)
    for subset in subsets:
        if subset['fact'] not in FACT_DETAIL_SUBSETS:
            self._module.fail_json(msg='Subset must be one of [%s], got %s' % (', '.join(sorted(FACT_DETAIL_SUBSETS)), subset['fact']))
        for valid_subset in frozenset(self.FACT_SUBSETS.keys()):
            if subset['fact'].startswith(valid_subset):
                runable_subsets.append((subset, valid_subset))
    return runable_subsets