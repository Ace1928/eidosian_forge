from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.cmdref.telemetry.telemetry import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import NxosCmdRef
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.telemetry.telemetry import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
The command generator when state is deleted
        :rtype: A list
        :returns: the commands necessary to remove the current configuration
                  of the provided objects
        