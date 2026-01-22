from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def is_device_valid(meraki, serial, data):
    """ Parse a list of devices for a serial and return True if it's in the list """
    for device in data:
        if device['serial'] == serial:
            return True
    return False