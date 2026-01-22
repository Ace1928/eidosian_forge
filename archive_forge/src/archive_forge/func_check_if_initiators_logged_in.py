from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def check_if_initiators_logged_in(self, initiators):
    """ Checks if any of the initiators is of type logged-in"""
    for item in initiators:
        initiator_details = utils.host.UnityHostInitiatorList.get(cli=self.unity._cli, initiator_id=item)._get_properties()
        if initiator_details['paths'][0] is not None and 'UnityHostInitiatorPathList' in initiator_details['paths'][0]:
            error_message = 'Removal operation cannot be done since host has logged in initiator(s)'
            LOG.error(error_message)
            self.module.fail_json(msg=error_message)