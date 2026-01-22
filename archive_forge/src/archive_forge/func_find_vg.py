from __future__ import absolute_import, division, print_function
import itertools
import os
from ansible.module_utils.basic import AnsibleModule
def find_vg(module, vg):
    if not vg:
        return None
    vgs_cmd = module.get_bin_path('vgs', True)
    dummy, current_vgs, dummy = module.run_command("%s --noheadings -o vg_name,pv_count,lv_count --separator ';'" % vgs_cmd, check_rc=True)
    vgs = parse_vgs(current_vgs)
    for test_vg in vgs:
        if test_vg['name'] == vg:
            this_vg = test_vg
            break
    else:
        this_vg = None
    return this_vg