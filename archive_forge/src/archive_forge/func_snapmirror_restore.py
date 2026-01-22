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
def snapmirror_restore(self):
    """
        restore SnapMirror based on relationship state
        """
    if self.use_rest:
        return self.snapmirror_restore_rest()
    options = {'destination-location': self.parameters['destination_path'], 'source-location': self.parameters['source_path']}
    if self.parameters.get('source_snapshot'):
        options['source-snapshot'] = self.parameters['source_snapshot']
    if self.parameters.get('clean_up_failure'):
        options['clean-up-failure'] = self.na_helper.get_value_for_bool(False, self.parameters['clean_up_failure'])
    snapmirror_restore = netapp_utils.zapi.NaElement.create_node_with_children('snapmirror-restore', **options)
    try:
        self.server.invoke_successfully(snapmirror_restore, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error restoring SnapMirror relationship: %s' % to_native(error), exception=traceback.format_exc())