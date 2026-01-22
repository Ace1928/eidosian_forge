from __future__ import absolute_import, division, print_function
import itertools
import os
from ansible.module_utils.basic import AnsibleModule
def reset_uuid_vg(module, vg):
    changed = False
    vgchange_cmd = module.get_bin_path('vgchange', True)
    vgchange_cmd_with_opts = [vgchange_cmd, '-u', vg]
    if module.check_mode:
        changed = True
    else:
        module.run_command(vgchange_cmd_with_opts, check_rc=True)
        changed = True
    return changed