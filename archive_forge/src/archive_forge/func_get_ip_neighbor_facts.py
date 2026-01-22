from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_ip_neighbor_facts(self):
    """ Get the 'facts' (the current configuration)

        :rtype: A dictionary
        :returns: The current configuration as a dictionary
        """
    facts, _warnings = Facts(self._module).get_facts(self.gather_subset, self.gather_network_resources)
    ip_neighbor_facts = facts['ansible_network_resources'].get('ip_neighbor')
    if not ip_neighbor_facts:
        requests = self.build_create_all_requests()
        try:
            edit_config(self._module, to_request(self._module, requests))
        except ConnectionError as exc:
            self._module.fail_json(msg=str(exc), code=exc.code)
        facts, _warnings = Facts(self._module).get_facts(self.gather_subset, self.gather_network_resources)
        ip_neighbor_facts = facts['ansible_network_resources'].get('ip_neighbor')
        if not ip_neighbor_facts:
            err_msg = 'IP neighbor module: get facts failed.'
            self._module.fail_json(msg=err_msg, code=500)
    return ip_neighbor_facts