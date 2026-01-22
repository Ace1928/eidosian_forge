from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url

    This is the structure of the 'PERMISSIONS' dictionary:

   "PERMISSIONS": {
                      "OWNER_U": "1",
                      "OWNER_M": "1",
                      "OWNER_A": "0",
                      "GROUP_U": "0",
                      "GROUP_M": "0",
                      "GROUP_A": "0",
                      "OTHER_U": "0",
                      "OTHER_M": "0",
                      "OTHER_A": "0"
                    }
    