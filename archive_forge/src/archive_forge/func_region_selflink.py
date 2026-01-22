from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
def region_selflink(name, params):
    if name is None:
        return
    url = 'https://compute.googleapis.com/compute/v1/projects/.*/regions/.*'
    if not re.match(url, name):
        name = 'https://compute.googleapis.com/compute/v1/projects/{project}/regions/%s'.format(**params) % name
    return name