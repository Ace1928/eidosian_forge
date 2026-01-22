from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def get_lvm_version(module):
    ver_cmd = module.get_bin_path('lvm', required=True)
    rc, out, err = module.run_command('%s version' % ver_cmd)
    if rc != 0:
        return None
    m = re.search('LVM version:\\s+(\\d+)\\.(\\d+)\\.(\\d+).*(\\d{4}-\\d{2}-\\d{2})', out)
    if not m:
        return None
    return mkversion(m.group(1), m.group(2), m.group(3))