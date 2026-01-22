from __future__ import absolute_import, division, print_function
import codecs
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_bytes, to_native, to_text
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def get_subset(self, gather_subset, version):
    """Method to get a single subset"""
    runable_subsets = set()
    exclude_subsets = set()
    usable_subsets = [key for key in self.info_subsets if version >= self.info_subsets[key]['min_version']]
    if 'help' in gather_subset:
        return usable_subsets
    for subset in gather_subset:
        if subset == 'all':
            runable_subsets.update(usable_subsets)
            return runable_subsets
        if subset.startswith('!'):
            subset = subset[1:]
            if subset == 'all':
                return set()
            exclude = True
        else:
            exclude = False
        if subset not in usable_subsets:
            if subset not in self.info_subsets.keys():
                self.module.fail_json(msg='Bad subset: %s' % subset)
            self.module.fail_json(msg='Remote system at version %s does not support %s' % (version, subset))
        if exclude:
            exclude_subsets.add(subset)
        else:
            runable_subsets.add(subset)
    if not runable_subsets:
        runable_subsets.update(usable_subsets)
    runable_subsets.difference_update(exclude_subsets)
    return runable_subsets