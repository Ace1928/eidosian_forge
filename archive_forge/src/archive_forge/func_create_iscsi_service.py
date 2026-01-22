from __future__ import absolute_import, division, print_function
import traceback
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_iscsi_service(self):
    """
        Create iscsi service and start if requested
        """
    if self.use_rest:
        return self.create_iscsi_service_rest()
    iscsi_service = netapp_utils.zapi.NaElement.create_node_with_children('iscsi-service-create', **{'start': 'true' if self.parameters.get('service_state', 'started') == 'started' else 'false'})
    try:
        self.server.invoke_successfully(iscsi_service, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as e:
        self.module.fail_json(msg='Error creating iscsi service: % s' % to_native(e), exception=traceback.format_exc())