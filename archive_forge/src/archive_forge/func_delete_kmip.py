from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def delete_kmip(module, array):
    """Delete existing KMIP object"""
    changed = True
    if not module.check_mode:
        res = array.delete_kmip(names=[module.params['name']])
        if res.status_code != 200:
            module.fail_json(msg='Failed to delete {0} KMIP object. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)