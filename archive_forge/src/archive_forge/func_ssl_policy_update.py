from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import time
def ssl_policy_update(module, request, response):
    auth = GcpSession(module, 'compute')
    auth.post(''.join(['https://compute.googleapis.com/compute/v1/', 'projects/{project}/global/targetSslProxies/{name}/setSslPolicy']).format(**module.params), {u'sslPolicy': replace_resource_dict(module.params.get(u'ssl_policy', {}), 'selfLink')})