from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
def collect_with_namespace(self, module=None, collected_facts=None):
    facts_dict = self.collect(module=module, collected_facts=collected_facts)
    if self.namespace:
        facts_dict = self._transform_dict_keys(facts_dict)
    return facts_dict