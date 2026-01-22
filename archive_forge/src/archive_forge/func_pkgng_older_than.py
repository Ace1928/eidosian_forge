from __future__ import absolute_import, division, print_function
from collections import defaultdict
import re
from ansible.module_utils.basic import AnsibleModule
def pkgng_older_than(module, pkgng_path, compare_version):
    rc, out, err = module.run_command([pkgng_path, '-v'])
    version = [int(x) for x in re.split('[\\._]', out)]
    i = 0
    new_pkgng = True
    while compare_version[i] == version[i]:
        i += 1
        if i == min(len(compare_version), len(version)):
            break
    else:
        if compare_version[i] > version[i]:
            new_pkgng = False
    return not new_pkgng