from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def get_fc_host_initiators_details(self, fc_host_initiators):
    """ Get the details of existing FC initiators in host"""
    fc_initiator_list = []
    for fc in fc_host_initiators:
        fc_initiator_details = self.unity.get_initiator(_id=fc.id)
        fc_path_list = []
        if fc_initiator_details.paths is not None:
            for path in fc_initiator_details.paths:
                fc_path_list.append({'id': path.id, 'is_logged_in': path.is_logged_in})
        fc_initiator_list.append({'id': fc_initiator_details.id, 'name': fc_initiator_details.initiator_id, 'paths': fc_path_list})
    return fc_initiator_list