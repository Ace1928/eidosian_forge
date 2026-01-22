from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
def find_collectors_for_platform(all_collector_classes, compat_platforms):
    found_collectors = set()
    found_collectors_names = set()
    for compat_platform in compat_platforms:
        platform_match = None
        for all_collector_class in all_collector_classes:
            platform_match = all_collector_class.platform_match(compat_platform)
            if not platform_match:
                continue
            primary_name = all_collector_class.name
            if primary_name not in found_collectors_names:
                found_collectors.add(all_collector_class)
                found_collectors_names.add(all_collector_class.name)
    return found_collectors