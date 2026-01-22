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
def snapmirror_delete_rest(self):
    """
        Delete SnapMirror relationship at destination cluster using REST
        """
    uuid = self.get_relationship_uuid(after_create=False)
    if uuid is None:
        self.module.fail_json(msg='Error in deleting SnapMirror: %s, unable to get UUID for the SnapMirror relationship.' % uuid)
    api = 'snapmirror/relationships'
    dummy, error = rest_generic.delete_async(self.rest_api, api, uuid)
    if error:
        msg = 'Error deleting SnapMirror: %s' % to_native(error)
        if self.previous_errors:
            msg += '.  Previous error(s): %s' % ' -- '.join(self.previous_errors)
        self.module.fail_json(msg=msg, exception=traceback.format_exc())