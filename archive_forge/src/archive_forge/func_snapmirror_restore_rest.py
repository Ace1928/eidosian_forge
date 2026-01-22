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
def snapmirror_restore_rest(self):
    """ snapmirror restore using rest """
    body = {'destination.path': self.parameters['destination_path'], 'source.path': self.parameters['source_path'], 'restore': 'true'}
    api = 'snapmirror/relationships'
    dummy, error = rest_generic.post_async(self.rest_api, api, body, timeout=120)
    if error:
        self.module.fail_json(msg='Error restoring SnapMirror: %s' % to_native(error), exception=traceback.format_exc())
    relationship_uuid = self.get_relationship_uuid()
    if relationship_uuid is None:
        self.module.fail_json(msg='Error restoring SnapMirror: unable to get UUID for the SnapMirror relationship.')
    body = {'source_snapshot': self.parameters['source_snapshot']} if self.parameters.get('source_snapshot') else {}
    api = 'snapmirror/relationships/%s/transfers' % relationship_uuid
    dummy, error = rest_generic.post_async(self.rest_api, api, body, timeout=60, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error restoring SnapMirror Transfer: %s' % to_native(error), exception=traceback.format_exc())