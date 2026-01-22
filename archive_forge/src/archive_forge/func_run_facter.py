from __future__ import (absolute_import, division, print_function)
import json
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.namespace import PrefixFactNamespace
from ansible.module_utils.facts.collector import BaseFactCollector
def run_facter(self, module, facter_path):
    rc, out, err = module.run_command(facter_path + ' --puppet --json')
    if rc != 0:
        rc, out, err = module.run_command(facter_path + ' --json')
    return (rc, out, err)