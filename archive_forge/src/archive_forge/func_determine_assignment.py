from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def determine_assignment(self, owned_partitions, own_spare_disks):
    """
        Determine which action to take
        return: dict containing lists of the disks/partitions to be assigned/reassigned
        """
    assignment = {'required_unassigned_partitions': [], 'required_partner_spare_disks': [], 'required_partner_spare_partitions': [], 'required_unassigned_disks': []}
    unassigned_partitions = self.get_partitions(container_type='owned', node=self.parameters['node'])
    required_partitions = self.parameters['partition_count'] - (len(owned_partitions) + len(own_spare_disks))
    if required_partitions > len(unassigned_partitions):
        assignment['required_unassigned_partitions'] = unassigned_partitions
        unassigned_disks = self.get_disks(container_type='spare')
        required_unassigned_disks = required_partitions - len(unassigned_partitions)
        if required_unassigned_disks > len(unassigned_disks):
            assignment['required_unassigned_disks'] = unassigned_disks
            required_partner_spare_partitions = required_unassigned_disks - len(unassigned_disks)
            partner_node_name = self.get_partner_node_name()
            if partner_node_name:
                partner_spare_partitions = self.get_partitions(container_type='spare', node=partner_node_name)
                partner_spare_disks = self.get_disks(container_type='spare', node=partner_node_name)
            else:
                partner_spare_partitions = []
                partner_spare_disks = []
            if required_partner_spare_partitions <= len(partner_spare_partitions) - self.parameters['min_spares']:
                assignment['required_partner_spare_partitions'] = partner_spare_partitions[0:required_partner_spare_partitions]
            elif len(partner_spare_disks) >= self.parameters['min_spares']:
                if required_partner_spare_partitions <= len(partner_spare_partitions):
                    assignment['required_partner_spare_partitions'] = partner_spare_partitions[0:required_partner_spare_partitions]
                else:
                    required_partner_spare_disks = required_partner_spare_partitions - len(partner_spare_partitions)
                    required_partner_spare_partitions_count = required_partner_spare_partitions - required_partner_spare_disks
                    assignment['required_partner_spare_partitions'] = partner_spare_partitions[0:required_partner_spare_partitions_count]
                    if required_partner_spare_disks > len(partner_spare_disks) - self.parameters['min_spares']:
                        self.module.fail_json(msg='Not enough partner spare disks or partner spare partitions remain to fulfill the request')
                    else:
                        assignment['required_partner_spare_disks'] = partner_spare_disks[0:required_partner_spare_disks]
            else:
                self.module.fail_json(msg='Not enough partner spare disks or partner spare partitions remain to fulfill the request')
        else:
            assignment['required_unassigned_disks'] = unassigned_disks[0:required_unassigned_disks]
    else:
        assignment['required_unassigned_partitions'] = unassigned_partitions[0:required_partitions]
    return assignment