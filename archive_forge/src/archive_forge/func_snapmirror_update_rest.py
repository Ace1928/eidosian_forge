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
def snapmirror_update_rest(self):
    """
        Perform an update on the relationship using POST on /snapmirror/relationships/{relationship.uuid}/transfers
        """
    uuid = self.get_relationship_uuid()
    if uuid is None:
        self.module.fail_json(msg='Error in updating SnapMirror relationship: unable to get UUID for the SnapMirror relationship.')
    api = 'snapmirror/relationships/%s/transfers' % uuid
    body = {}
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error updating SnapMirror relationship: %s:' % to_native(error), exception=traceback.format_exc())