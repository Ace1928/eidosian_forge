from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils._mount import ismount
import re
def unmount_fs(module, filesystem):
    """ Unmount a file system."""
    unmount_cmd = module.get_bin_path('unmount', True)
    if not module.check_mode:
        rc, unmount_out, err = module.run_command([unmount_cmd, filesystem])
        if rc != 0:
            module.fail_json(msg='Failed to run unmount. Error message: %s' % err)
        else:
            changed = True
            msg = 'File system %s unmounted.' % filesystem
            return (changed, msg)
    else:
        changed = True
        msg = ''
        return (changed, msg)