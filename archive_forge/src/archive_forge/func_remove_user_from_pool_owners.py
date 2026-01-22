from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def remove_user_from_pool_owners(user, pool):
    """ Remove user from pool owners """
    changed = False
    pool_fields = pool.get_fields(from_cache=True, raw_value=True)
    pool_owners = pool_fields.get('owners', [])
    try:
        pool_owners.remove(user)
        pool.set_owners(pool_owners)
        changed = True
    except ValueError:
        pass
    return changed