from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_image_package_delete_rest(self):
    api = 'cluster/software/packages'
    dummy, error = rest_generic.delete_async(self.rest_api, api, self.parameters['package_version'])
    if error:
        self.module.fail_json(msg='Error deleting cluster software package for %s: %s' % (self.parameters['package_version'], error))