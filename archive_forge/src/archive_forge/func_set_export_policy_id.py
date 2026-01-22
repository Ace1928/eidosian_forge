from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def set_export_policy_id(self):
    """
        Fetch export-policy id
        :param:
            name : Name of the export-policy

        :return: Set self.policy_id
        """
    if self.policy_id is not None:
        return
    if self.use_rest:
        return self.set_export_policy_id_rest()
    export_policy_iter = netapp_utils.zapi.NaElement('export-policy-get-iter')
    attributes = {'query': {'export-policy-info': {'policy-name': self.parameters['name'], 'vserver': self.parameters['vserver']}}}
    export_policy_iter.translate_struct(attributes)
    try:
        result = self.server.invoke_successfully(export_policy_iter, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error getting export policy %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) == 1:
        self.policy_id = self.na_helper.safe_get(result, ['attributes-list', 'export-policy-info', 'policy-id'])
        if self.policy_id is None:
            self.module.fail_json(msg='Error getting export policy id for %s: got: %s.' % (self.parameters['name'], result.to_string()))