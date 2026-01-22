from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def remove_load_balancer(module, oneandone_conn):
    """
    Removes a load_balancer.

    module : AnsibleModule object
    oneandone_conn: authenticated oneandone object
    """
    try:
        lb_id = module.params.get('name')
        load_balancer_id = get_load_balancer(oneandone_conn, lb_id)
        if module.check_mode:
            if load_balancer_id is None:
                _check_mode(module, False)
            _check_mode(module, True)
        load_balancer = oneandone_conn.delete_load_balancer(load_balancer_id)
        changed = True if load_balancer else False
        return (changed, {'id': load_balancer['id'], 'name': load_balancer['name']})
    except Exception as ex:
        module.fail_json(msg=str(ex))