from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def set_disk_count(self, current, modify):
    if modify.get('disk_count'):
        if int(modify['disk_count']) < int(current['disk_count']):
            self.module.fail_json(msg='Error: specified disk_count is less than current disk_count. Only adding disks is allowed.')
        else:
            modify['disk_count'] = modify['disk_count'] - current['disk_count']