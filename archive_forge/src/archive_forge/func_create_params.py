from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
def create_params(self):
    new_object_params = {}
    if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
        new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
    if self.new_object.get('ssid') is not None or self.new_object.get('ssid') is not None:
        new_object_params['ssid'] = self.new_object.get('ssid') or self.new_object.get('ssid')
    if self.new_object.get('identity') is not None or self.new_object.get('identity') is not None:
        new_object_params['identity'] = self.new_object.get('identity') or self.new_object.get('identity')
    if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
        new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
    return new_object_params