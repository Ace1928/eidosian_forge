from __future__ import (absolute_import, division, print_function)
import json
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.namespace import PrefixFactNamespace
from ansible.module_utils.facts.collector import BaseFactCollector
def run_ohai(self, module, ohai_path):
    rc, out, err = module.run_command(ohai_path)
    return (rc, out, err)