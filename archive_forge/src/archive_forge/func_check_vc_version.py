from __future__ import absolute_import, division, print_function
import_profile:
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
import json
import time
def check_vc_version(self):
    if LooseVersion(self.content.about.version) < LooseVersion('7'):
        self.module.fail_json(msg='vCenter version is less than 7.0.0 Please specify vCenter with version greater than or equal to 7.0.0')