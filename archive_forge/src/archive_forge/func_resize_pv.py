from __future__ import absolute_import, division, print_function
import itertools
import os
from ansible.module_utils.basic import AnsibleModule
def resize_pv(module, device):
    changed = False
    pvresize_cmd = module.get_bin_path('pvresize', True)
    dev_size, pv_size, pe_start, vg_extent_size = get_pv_values_for_resize(module=module, device=device)
    if dev_size - (pe_start + pv_size) > vg_extent_size:
        if module.check_mode:
            changed = True
        else:
            rc, out, err = module.run_command([pvresize_cmd, device])
            dummy, new_pv_size, dummy, dummy = get_pv_values_for_resize(module=module, device=device)
            if pv_size == new_pv_size:
                module.fail_json(msg='Failed executing pvresize command.', rc=rc, err=err, out=out)
            else:
                changed = True
    return changed