from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def pmem_init_env(self):
    if self.namespace is None or (self.namespace and self.namespace_append is False):
        self.pmem_remove_namespaces()
    if self.namespace is None:
        self.pmem_delete_goal()