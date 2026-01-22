from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def start_volume_clone_split(self):
    """
        Starts a volume clone split
        """
    if self.use_rest:
        return self.start_volume_clone_split_rest()
    clone_obj = netapp_utils.zapi.NaElement('volume-clone-split-start')
    clone_obj.add_new_child('volume', self.parameters['name'])
    try:
        self.vserver.invoke_successfully(clone_obj, True)
    except netapp_utils.zapi.NaApiError as exc:
        self.module.fail_json(msg='Error starting volume clone split: %s: %s' % (self.parameters['name'], to_native(exc)))