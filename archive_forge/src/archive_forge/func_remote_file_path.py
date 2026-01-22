from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import os
def remote_file_path(module):
    if module.params['action'] == 'download':
        return module.params['src']
    elif module.params['action'] == 'delete':
        return module.params['src']
    else:
        return module.params['dest']