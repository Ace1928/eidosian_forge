from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def get_list_unmapped_initiators(self, initiators, host_id=None):
    """ Get the list of those initiators which are
            not mapped to any host"""
    unmapped_initiators = []
    for id in initiators:
        initiator_details = utils.host.UnityHostInitiatorList.get(cli=self.unity._cli, initiator_id=id)._get_properties()
        ' if an already existing initiator is passed along with an\n                unmapped initiator'
        if None in initiator_details['parent_host']:
            unmapped_initiators.append(initiator_details['initiator_id'][0])
        elif not initiator_details['parent_host']:
            unmapped_initiators.append(id)
        else:
            error_message = 'Initiator ' + id + ' mapped to another Host.'
            LOG.error(error_message)
            self.module.fail_json(msg=error_message)
    return unmapped_initiators