from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
def find_unresolved_requires(collector_names, all_fact_subsets):
    """Find any collector names that have unresolved requires

    Returns a list of collector names that correspond to collector
    classes whose .requires_facts() are not in collector_names.
    """
    unresolved = set()
    for collector_name in collector_names:
        required_facts = _get_requires_by_collector_name(collector_name, all_fact_subsets)
        for required_fact in required_facts:
            if required_fact not in collector_names:
                unresolved.add(required_fact)
    return unresolved