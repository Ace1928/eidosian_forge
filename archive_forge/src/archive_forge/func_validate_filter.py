from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def validate_filter(self, filter_dict):
    """ Validate given filter_dict """
    is_invalid_filter = self.filter_keys != sorted(list(filter_dict))
    if is_invalid_filter:
        msg = "Filter should have all keys: '{0}'".format(', '.join(self.filter_keys))
        LOG.error(msg)
        self.module.fail_json(msg=msg)
    is_invalid_filter = [filter_dict[i] is None for i in filter_dict]
    if True in is_invalid_filter:
        msg = "Filter keys: '{0}' cannot be None".format(self.filter_keys)
        LOG.error(msg)
        self.module.fail_json(msg=msg)