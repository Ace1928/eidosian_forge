from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def rename_export_policy(self):
    """
        Rename the export-policy.
        """
    export_policy_rename = netapp_utils.zapi.NaElement.create_node_with_children('export-policy-rename', **{'policy-name': self.parameters['from_name'], 'new-policy-name': self.parameters['name']})
    try:
        self.server.invoke_successfully(export_policy_rename, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error on renaming export-policy %s:%s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())