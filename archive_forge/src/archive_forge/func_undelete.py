from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
def undelete(module, link, etag):
    auth = GcpSession(module, 'iam')
    return return_if_object(module, auth.post(link + ':undelete', {'etag': etag}))