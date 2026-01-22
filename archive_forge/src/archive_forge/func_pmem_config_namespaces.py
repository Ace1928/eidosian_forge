from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def pmem_config_namespaces(self, namespace):
    command = ['create-namespace', '-m', namespace['mode']]
    if namespace['type']:
        command += ['-t', namespace['type']]
    if 'size_byte' in namespace:
        command += ['-s', namespace['size_byte']]
    self.pmem_run_ndctl(command)
    return None