from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def delete_nulls(h):
    """ Remove null entries from a hash

    Returns:
        a hash without nulls
    """
    if isinstance(h, list):
        return [delete_nulls(i) for i in h]
    if isinstance(h, dict):
        return dict(((k, delete_nulls(v)) for k, v in h.items() if v is not None))
    return h