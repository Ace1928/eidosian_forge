from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
from re import sub
def meraki_get_radio_settings(meraki):
    """Query the Meraki API for the current radio settings."""
    path = meraki.construct_path('get_one', custom={'serial': meraki.params['serial']})
    return meraki.request(path, method='GET')