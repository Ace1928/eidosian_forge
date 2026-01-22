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
def snapmirror_abort(self):
    """
        Abort a SnapMirror relationship in progress
        """
    if self.use_rest:
        return self.snapmirror_abort_rest()
    options = {'destination-location': self.parameters['destination_path']}
    snapmirror_abort = netapp_utils.zapi.NaElement.create_node_with_children('snapmirror-abort', **options)
    try:
        self.server.invoke_successfully(snapmirror_abort, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error aborting SnapMirror relationship: %s' % to_native(error), exception=traceback.format_exc())