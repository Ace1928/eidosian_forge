from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
def get_params_by_id(self, name=None, id=None):
    new_object_params = {}
    if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
        new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
    if self.new_object.get('portId') is not None or self.new_object.get('port_id') is not None:
        new_object_params['portId'] = self.new_object.get('portId') or self.new_object.get('port_id')
    return new_object_params