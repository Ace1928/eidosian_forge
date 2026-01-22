from __future__ import (absolute_import, division, print_function)
import json
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.namespace import PrefixFactNamespace
from ansible.module_utils.facts.collector import BaseFactCollector
def get_facter_output(self, module):
    facter_path = self.find_facter(module)
    if not facter_path:
        return None
    rc, out, err = self.run_facter(module, facter_path)
    if rc != 0:
        return None
    return out