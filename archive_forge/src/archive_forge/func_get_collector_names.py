from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
def get_collector_names(valid_subsets=None, minimal_gather_subset=None, gather_subset=None, aliases_map=None, platform_info=None):
    """return a set of FactCollector names based on gather_subset spec.

    gather_subset is a spec describing which facts to gather.
    valid_subsets is a frozenset of potential matches for gather_subset ('all', 'network') etc
    minimal_gather_subsets is a frozenset of matches to always use, even for gather_subset='!all'
    """
    gather_subset = gather_subset or ['all']
    valid_subsets = valid_subsets or frozenset()
    minimal_gather_subset = minimal_gather_subset or frozenset()
    aliases_map = aliases_map or defaultdict(set)
    additional_subsets = set()
    exclude_subsets = set()
    gather_subset_with_min = ['min']
    gather_subset_with_min.extend(gather_subset)
    explicitly_added = set()
    for subset in gather_subset_with_min:
        subset_id = subset
        if subset_id == 'min':
            additional_subsets.update(minimal_gather_subset)
            continue
        if subset_id == 'all':
            additional_subsets.update(valid_subsets)
            continue
        if subset_id.startswith('!'):
            subset = subset[1:]
            if subset == 'min':
                exclude_subsets.update(minimal_gather_subset)
                continue
            if subset == 'all':
                exclude_subsets.update(valid_subsets - minimal_gather_subset)
                continue
            exclude = True
        else:
            exclude = False
        if exclude:
            exclude_subsets.update(aliases_map.get(subset, set()))
            exclude_subsets.add(subset)
        else:
            if subset_id not in valid_subsets:
                raise TypeError("Bad subset '%s' given to Ansible. gather_subset options allowed: all, %s" % (subset, ', '.join(sorted(valid_subsets))))
            explicitly_added.add(subset)
            additional_subsets.add(subset)
    if not additional_subsets:
        additional_subsets.update(valid_subsets)
    additional_subsets.difference_update(exclude_subsets - explicitly_added)
    return additional_subsets