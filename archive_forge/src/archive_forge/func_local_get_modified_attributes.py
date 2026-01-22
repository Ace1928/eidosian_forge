from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def local_get_modified_attributes(self, current):
    modify = self.na_helper.get_modified_attributes(current, self.parameters)
    if current and 'external' in self.parameters and (not self.na_helper.safe_get(modify, ['external', 'servers'])):
        current_servers = self.na_helper.safe_get(current, ['external', 'servers'])
        desired_servers = self.na_helper.safe_get(self.parameters, ['external', 'servers'])
        if current_servers != desired_servers:
            if 'external' not in modify:
                modify['external'] = {}
            modify['external']['servers'] = desired_servers
            self.na_helper.changed = True
    return modify