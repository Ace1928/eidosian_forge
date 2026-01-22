from __future__ import absolute_import, division, print_function
import time
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_quota_status_or_volume_id_rest(self, get_volume=None):
    """
        Get the status info on or off
        """
    if not self.use_rest:
        return self.get_quota_status()
    api = 'storage/volumes'
    params = {'name': self.parameters['volume'], 'svm.name': self.parameters['vserver'], 'fields': 'quota.state,uuid'}
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        msg = 'volume uuid' if get_volume else 'quota status info'
        self.module.fail_json(msg='Error on getting %s: %s' % (msg, error))
    if record:
        return record['uuid'] if get_volume else record['quota']['state']
    self.module.fail_json(msg='Error: Volume %s in SVM %s does not exist' % (self.parameters['volume'], self.parameters['vserver']))