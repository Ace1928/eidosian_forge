from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest
import json
import re
def name_pattern(name, module):
    if name is None:
        return
    regex = 'projects/.*/repos/.*'
    if not re.match(regex, name):
        name = 'projects/{project}/repos/{name}'.format(**module.params)
    return name