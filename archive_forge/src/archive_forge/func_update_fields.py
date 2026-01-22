from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
def update_fields(module, request, response):
    if response.get('labels') != request.get('labels'):
        label_fingerprint_update(module, request, response)
    if response.get('sizeGb') != request.get('sizeGb'):
        size_gb_update(module, request, response)