from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils._mount import ismount
import re
def mount_fs(module, filesystem):
    """ Mount a file system. """
    mount_cmd = module.get_bin_path('mount', True)
    if not module.check_mode:
        rc, mount_out, err = module.run_command([mount_cmd, filesystem])
        if rc != 0:
            module.fail_json(msg='Failed to run mount. Error message: %s' % err)
        else:
            changed = True
            msg = 'File system %s mounted.' % filesystem
            return (changed, msg)
    else:
        changed = True
        msg = ''
        return (changed, msg)