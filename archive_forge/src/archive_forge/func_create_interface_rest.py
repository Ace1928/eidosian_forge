from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def create_interface_rest(self, body):
    """ calling REST to create interface """
    query = {'return_records': 'true'}
    records, error = rest_generic.post_async(self.rest_api, self.get_net_int_api(), body, query)
    if error:
        self.module.fail_json(msg='Error creating interface %s: %s' % (self.parameters['interface_name'], to_native(error)), exception=traceback.format_exc())
    return records