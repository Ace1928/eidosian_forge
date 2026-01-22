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
def modify_lun_rest(self, modify):
    local_modify = modify.copy()
    if self.uuid is None:
        self.module.fail_json(msg='Error modifying LUN %s: UUID not found' % self.parameters['name'])
    api = 'storage/luns'
    body = {}
    if local_modify.get('space_reserve') is not None:
        body['space.guarantee.requested'] = local_modify.pop('space_reserve')
    if local_modify.get('space_allocation') is not None:
        body['space.scsi_thin_provisioning_support_enabled'] = local_modify.pop('space_allocation')
    if local_modify.get('comment') is not None:
        body['comment'] = local_modify.pop('comment')
    if local_modify.get('qos_policy_group') is not None:
        body['qos_policy.name'] = local_modify.pop('qos_policy_group')
    if local_modify != {}:
        self.module.fail_json(msg='Error modifying LUN %s: Unknown parameters: %s' % (self.parameters['name'], local_modify))
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, body)
    if error:
        self.module.fail_json(msg='Error modifying LUN %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())