from __future__ import absolute_import, division, print_function
import collections
import os
from copy import deepcopy
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import Version
def copy_key_to_node(self, base64keyfile):
    """Copy key to IOS-XR node. We use SFTP because older IOS-XR versions don't handle SCP very well."""
    if self._module.params['aggregate']:
        name = 'aggregate'
    else:
        name = self._module.params['name']
    src = base64keyfile
    dst = '/harddisk:/publickey_%s.b64' % name
    copy_file(self._module, src, dst)