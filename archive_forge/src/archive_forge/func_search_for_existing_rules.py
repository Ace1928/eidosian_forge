from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.check_point.mgmt.plugins.module_utils.checkpoint import (
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.check_point.mgmt.plugins.modules.cp_mgmt_threat_layers import (
def search_for_existing_rules(self, conn_request, api_call_object, search_payload=None, state=None):
    result = conn_request.post(api_call_object, state, data=search_payload)
    return result