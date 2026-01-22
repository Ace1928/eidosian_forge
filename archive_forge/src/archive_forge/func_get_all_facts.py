from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.facts.namespace import PrefixFactNamespace
from ansible.module_utils.facts import default_collectors
from ansible.module_utils.facts import ansible_collector
def get_all_facts(module):
    """compat api for ansible 2.2/2.3 module_utils.facts.get_all_facts method

    Expects module to be an instance of AnsibleModule, with a 'gather_subset' param.

    returns a dict mapping the bare fact name ('default_ipv4' with no 'ansible_' namespace) to
    the fact value."""
    gather_subset = module.params['gather_subset']
    return ansible_facts(module, gather_subset=gather_subset)