from __future__ import absolute_import, division, print_function
import time
def parse_subnet(self):
    if isinstance(self.subnet, dict):
        if 'virtual_network_name' not in self.subnet or 'name' not in self.subnet:
            self.fail('Subnet dict must contains virtual_network_name and name')
        if 'resource_group' not in self.subnet:
            self.subnet['resource_group'] = self.resource_group
        subnet_id = self.get_subnet()
    else:
        subnet_id = self.subnet
    return subnet_id