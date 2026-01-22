from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
def labels_update(module, request, response):
    auth = GcpSession(module, 'compute')
    auth.post(''.join(['https://compute.googleapis.com/compute/v1/', 'projects/{project}/global/snapshots/{name}/setLabels']).format(**module.params), {u'labels': module.params.get('labels'), u'labelFingerprint': response.get('labelFingerprint')})