from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_syslog(module, blade):
    """Delete Syslog Server"""
    changed = False
    try:
        server = blade.syslog.list_syslog_servers(names=[module.params['name']])
    except Exception:
        server = None
    if server:
        changed = True
        if not module.check_mode:
            try:
                blade.syslog.delete_syslog_servers(names=[module.params['name']])
                changed = True
            except Exception:
                module.fail_json(msg='Failed to remove syslog server: {0}'.format(module.params['name']))
    module.exit_json(changed=changed)