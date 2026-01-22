from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def get_undo_pod(module, array):
    """Return Undo Pod or None"""
    try:
        return array.get_pod(module.params['name'] + '.undo-demote', pending_only=True)
    except Exception:
        return None