from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.logging_global import (
def operation_rep(self, params):
    op_val = dict()
    for k, val in iteritems(params):
        if k in ['console', 'global_params']:
            mod_val = deepcopy(val)
            op_val.update(self.flatten_facility({k: mod_val}))
        elif k in ['files', 'hosts', 'users']:
            for m, n in iteritems(val):
                mod_n = deepcopy(n)
                if mod_n.get('archive'):
                    del mod_n['archive']
                if mod_n.get('facilities'):
                    del mod_n['facilities']
                if mod_n.get('port'):
                    del mod_n['port']
                tm = self.flatten_facility({k: {m: mod_n}})
                op_val.update(tm)
    return op_val