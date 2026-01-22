from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_create_body(self):
    """
        It gathers the required information for snapmirror create
        """
    initialized = False
    body = {'source': self.na_helper.filter_out_none_entries(self.parameters['source_endpoint']), 'destination': self.na_helper.filter_out_none_entries(self.parameters['destination_endpoint'])}
    if self.na_helper.safe_get(self.parameters, ['create_destination', 'enabled']):
        body['create_destination'] = self.na_helper.filter_out_none_entries(self.parameters['create_destination'])
        if self.parameters['initialize']:
            body['state'] = self.set_initialization_state()
            initialized = True
    if self.na_helper.safe_get(self.parameters, ['policy']) is not None:
        body['policy'] = {'name': self.parameters['policy']}
    if self.na_helper.safe_get(self.parameters, ['schedule']) is not None:
        body['transfer_schedule'] = {'name': self.string_or_none(self.parameters['schedule'])}
    if self.parameters.get('identity_preservation'):
        body['identity_preservation'] = self.parameters['identity_preservation']
    return (body, initialized)