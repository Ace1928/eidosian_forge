from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
def build_dep_data(collector_names, all_fact_subsets):
    dep_map = defaultdict(set)
    for collector_name in collector_names:
        collector_deps = set()
        for collector in all_fact_subsets[collector_name]:
            for dep in collector.required_facts:
                collector_deps.add(dep)
        dep_map[collector_name] = collector_deps
    return dep_map