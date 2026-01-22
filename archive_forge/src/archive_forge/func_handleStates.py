from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.logging_global import (
def handleStates(self, want=None, have=None):
    stateparsers = ['syslog.state', 'console.state', 'global_params.state', 'global_params.archive.state', 'files.archive.state']
    for par in stateparsers:
        op = get_from_dict(want, par)
        if op == 'enabled':
            self.addcmd(want, par)
        elif op == 'disabled':
            self.addcmd(want, par, True)
            break