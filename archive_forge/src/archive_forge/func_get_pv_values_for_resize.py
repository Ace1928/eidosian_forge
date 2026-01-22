from __future__ import absolute_import, division, print_function
import itertools
import os
from ansible.module_utils.basic import AnsibleModule
def get_pv_values_for_resize(module, device):
    pvdisplay_cmd = module.get_bin_path('pvdisplay', True)
    pvdisplay_ops = ['--units', 'b', '--columns', '--noheadings', '--nosuffix', '--separator', ';', '-o', 'dev_size,pv_size,pe_start,vg_extent_size']
    pvdisplay_cmd_device_options = [pvdisplay_cmd, device] + pvdisplay_ops
    dummy, pv_values, dummy = module.run_command(pvdisplay_cmd_device_options, check_rc=True)
    values = pv_values.strip().split(';')
    dev_size = int(values[0])
    pv_size = int(values[1])
    pe_start = int(values[2])
    vg_extent_size = int(values[3])
    return (dev_size, pv_size, pe_start, vg_extent_size)