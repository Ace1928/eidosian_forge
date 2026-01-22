from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def wrap_fail_json(self, msg, exception=None):
    for issue in self.issues:
        self.module.warn(issue)
    if self.volume_created:
        msg = 'Volume created with success, with missing attributes: %s' % msg
    self.module.fail_json(msg=msg, exception=exception)