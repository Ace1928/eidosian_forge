from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def pmem_remove_namespaces(self):
    command = ['list', '-N']
    out = self.pmem_run_ndctl(command)
    if not out:
        return
    namespaces = json.loads(out)
    for ns in namespaces:
        command = ['disable-namespace', ns['dev']]
        self.pmem_run_ndctl(command)
        command = ['destroy-namespace', ns['dev']]
        self.pmem_run_ndctl(command)
    return