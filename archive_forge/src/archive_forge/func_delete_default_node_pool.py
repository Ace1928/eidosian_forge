from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
def delete_default_node_pool(module):
    auth = GcpSession(module, 'container')
    link = 'https://container.googleapis.com/v1/projects/%s/locations/%s/clusters/%s/nodePools/default-pool' % (module.params['project'], module.params['location'], module.params['name'])
    return wait_for_operation(module, auth.delete(link))