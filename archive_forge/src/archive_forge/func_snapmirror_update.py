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
def snapmirror_update(self, relationship_type):
    """
        Update data in destination endpoint
        """
    if self.use_rest:
        return self.snapmirror_update_rest()
    zapi = 'snapmirror-update'
    options = {'destination-location': self.parameters['destination_path']}
    if relationship_type == 'load_sharing':
        zapi = 'snapmirror-update-ls-set'
        options = {'source-location': self.parameters['source_path']}
    snapmirror_update = netapp_utils.zapi.NaElement.create_node_with_children(zapi, **options)
    try:
        self.server.invoke_successfully(snapmirror_update, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error updating SnapMirror: %s' % to_native(error), exception=traceback.format_exc())