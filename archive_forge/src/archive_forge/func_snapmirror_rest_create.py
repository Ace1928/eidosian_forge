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
def snapmirror_rest_create(self):
    """
        Create a SnapMirror relationship using REST
        """
    body, initialized = self.get_create_body()
    api = 'snapmirror/relationships'
    dummy, error = rest_generic.post_async(self.rest_api, api, body, timeout=120)
    if error:
        self.module.fail_json(msg='Error creating SnapMirror: %s' % to_native(error), exception=traceback.format_exc())
    if self.parameters['initialize']:
        if initialized:
            self.wait_for_idle_status()
        else:
            self.snapmirror_initialize()