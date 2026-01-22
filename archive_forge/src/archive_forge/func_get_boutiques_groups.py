import os
import sys
import simplejson as json
from ..scripts.instance import import_module
def get_boutiques_groups(input_traits):
    """
    Returns a list of dictionaries containing Boutiques groups for the mutually
    exclusive Nipype inputs.
    """
    desc_groups = []
    mutex_input_sets = []
    for name, spec in input_traits:
        if spec.xor is not None:
            group_members = set([name] + list(spec.xor))
            if group_members not in mutex_input_sets:
                mutex_input_sets.append(group_members)
    for i, inp_set in enumerate(mutex_input_sets, 1):
        desc_groups.append({'id': 'mutex_group' + ('_' + str(i) if i != 1 else ''), 'name': 'Mutex group' + (' ' + str(i) if i != 1 else ''), 'members': list(inp_set), 'mutually-exclusive': True})
    return desc_groups