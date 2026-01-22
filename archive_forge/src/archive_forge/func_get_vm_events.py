from __future__ import absolute_import, division, print_function
import re
import time
import string
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils._text import to_text, to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def get_vm_events(self, vm, eventTypeIdList):
    byEntity = vim.event.EventFilterSpec.ByEntity(entity=vm, recursion='self')
    filterSpec = vim.event.EventFilterSpec(entity=byEntity, eventTypeId=eventTypeIdList)
    eventManager = self.content.eventManager
    return eventManager.QueryEvent(filterSpec)