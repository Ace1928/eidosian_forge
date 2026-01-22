from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def update_public_ip(module, oneandone_conn):
    """
    Update a public IP

    module : AnsibleModule object
    oneandone_conn: authenticated oneandone object

    Returns a dictionary containing a 'changed' attribute indicating whether
    any public IP was changed.
    """
    reverse_dns = module.params.get('reverse_dns')
    public_ip_id = module.params.get('public_ip_id')
    wait = module.params.get('wait')
    wait_timeout = module.params.get('wait_timeout')
    wait_interval = module.params.get('wait_interval')
    public_ip = get_public_ip(oneandone_conn, public_ip_id, True)
    if public_ip is None:
        _check_mode(module, False)
        module.fail_json(msg='public IP %s not found.' % public_ip_id)
    try:
        _check_mode(module, True)
        public_ip = oneandone_conn.modify_public_ip(ip_id=public_ip['id'], reverse_dns=reverse_dns)
        if wait:
            wait_for_resource_creation_completion(oneandone_conn, OneAndOneResources.public_ip, public_ip['id'], wait_timeout, wait_interval)
            public_ip = oneandone_conn.get_public_ip(public_ip['id'])
        changed = True if public_ip else False
        return (changed, public_ip)
    except Exception as e:
        module.fail_json(msg=str(e))