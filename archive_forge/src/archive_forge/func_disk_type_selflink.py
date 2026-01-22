from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
def disk_type_selflink(name, params):
    if name is None:
        return
    url = 'https://compute.googleapis.com/compute/v1/projects/.*/zones/.*/diskTypes/.*'
    if not re.match(url, name):
        name = 'https://compute.googleapis.com/compute/v1/projects/{project}/zones/{zone}/diskTypes/%s'.format(**params) % name
    return name