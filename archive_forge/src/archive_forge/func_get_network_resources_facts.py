from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.network import (
def get_network_resources_facts(self, facts_resource_obj_map, resource_facts_type=None, data=None):
    """
        :param fact_resource_subsets:
        :param data: previously collected configuration
        :return:
        """
    if not resource_facts_type:
        resource_facts_type = self._gather_network_resources
    restorun_subsets = self.gen_runable(resource_facts_type, frozenset(facts_resource_obj_map.keys()), resource_facts=True)
    if restorun_subsets:
        self.ansible_facts['ansible_net_gather_network_resources'] = list(restorun_subsets)
        instances = list()
        for key in restorun_subsets:
            fact_cls_obj = facts_resource_obj_map.get(key)
            if fact_cls_obj:
                instances.append(fact_cls_obj(self._module))
            else:
                self._warnings.extend(["network resource fact gathering for '%s' is not supported" % key])
        for inst in instances:
            try:
                inst.populate_facts(self._connection, self.ansible_facts, data)
            except Exception as exc:
                self._module.fail_json(msg=to_text(exc))