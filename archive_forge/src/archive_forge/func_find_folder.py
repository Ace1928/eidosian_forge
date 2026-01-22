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
def find_folder(self, searchpath):
    """ Walk inventory objects one position of the searchpath at a time """
    paths = [x.replace('/', '') for x in searchpath.split('/')]
    paths_total = len(paths) - 1
    position = 0
    root = self.content.rootFolder
    while root and position <= paths_total:
        change = False
        if hasattr(root, 'childEntity'):
            for child in root.childEntity:
                if child.name == paths[position]:
                    root = child
                    position += 1
                    change = True
                    break
        elif isinstance(root, vim.Datacenter):
            if hasattr(root, 'vmFolder'):
                if root.vmFolder.name == paths[position]:
                    root = root.vmFolder
                    position += 1
                    change = True
        else:
            root = None
        if not change:
            root = None
    return root