from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def get_bfd_interfaces_facts(self, data=None):
    """Get the 'facts' (the current configuration)

        :returns: A list of interface configs and a platform string
        """
    if self.state not in self.ACTION_STATES:
        self.gather_subset = ['!all', '!min']
    facts, _warnings = Facts(self._module).get_facts(self.gather_subset, self.gather_network_resources, data=data)
    bfd_interfaces_facts = facts['ansible_network_resources'].get('bfd_interfaces', [])
    platform = facts.get('ansible_net_platform', '')
    return (bfd_interfaces_facts, platform)