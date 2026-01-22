from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import (
def copy_file_to_node(module):
    """Copy config file to IOS-XR node. We use SFTP because older IOS-XR versions don't handle SCP very well."""
    file = tempfile.NamedTemporaryFile('wb', delete=False)
    src = os.path.realpath(file.name)
    file.write(to_bytes(module.params['src'], errors='surrogate_or_strict'))
    file.close()
    dst = '/harddisk:/ansible_config.txt'
    copy_file(module, src, dst, 'sftp')
    os.remove(src)
    return True