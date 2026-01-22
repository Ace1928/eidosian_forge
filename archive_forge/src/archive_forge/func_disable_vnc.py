from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def disable_vnc(module, array, app):
    """Disable VNC port"""
    changed = False
    if app.vnc_enabled:
        changed = True
        if not module.check_mode:
            res = array.patch_apps(names=[module.params['name']], app=App(vnc_enabled=False))
            if res.status_code != 200:
                module.fail_json(msg='Disabling VNC for {0} failed. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)