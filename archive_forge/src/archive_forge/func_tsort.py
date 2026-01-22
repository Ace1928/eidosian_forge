from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
def tsort(dep_map):
    sorted_list = []
    unsorted_map = dep_map.copy()
    while unsorted_map:
        acyclic = False
        for node, edges in list(unsorted_map.items()):
            for edge in edges:
                if edge in unsorted_map:
                    break
            else:
                acyclic = True
                del unsorted_map[node]
                sorted_list.append((node, edges))
        if not acyclic:
            raise CycleFoundInFactDeps('Unable to tsort deps, there was a cycle in the graph. sorted=%s' % sorted_list)
    return sorted_list