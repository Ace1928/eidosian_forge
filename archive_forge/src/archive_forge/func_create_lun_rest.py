from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_lun_rest(self):
    name = self.create_lun_path_rest()
    api = 'storage/luns'
    body = {'svm.name': self.parameters['vserver'], 'name': name}
    if self.parameters.get('flexvol_name') is not None:
        body['location.volume.name'] = self.parameters['flexvol_name']
    if self.parameters.get('qtree_name') is not None:
        body['location.qtree.name'] = self.parameters['qtree_name']
    if self.parameters.get('os_type') is not None:
        body['os_type'] = self.parameters['os_type']
    if self.parameters.get('size') is not None:
        body['space.size'] = self.parameters['size']
    if self.parameters.get('space_reserve') is not None:
        body['space.guarantee.requested'] = self.parameters['space_reserve']
    if self.parameters.get('space_allocation') is not None:
        body['space.scsi_thin_provisioning_support_enabled'] = self.parameters['space_allocation']
    if self.parameters.get('comment') is not None:
        body['comment'] = self.parameters['comment']
    if self.parameters.get('qos_policy_group') is not None:
        body['qos_policy.name'] = self.parameters['qos_policy_group']
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating LUN %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())