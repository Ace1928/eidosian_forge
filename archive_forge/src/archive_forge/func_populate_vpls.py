from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.ironware.ironware import run_commands
from ansible_collections.community.network.plugins.module_utils.network.ironware.ironware import ironware_argument_spec, check_args
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def populate_vpls(self, vpls):
    facts = dict()
    for key, value in iteritems(vpls):
        vpls = dict()
        vpls['endpoints'] = self.parse_vpls_endpoints(value)
        vpls['vc-id'] = self.parse_vpls_vcid(value)
        facts[key] = vpls
    return facts