from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError
from ansible_collections.kubernetes.core.plugins.module_utils.common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
def format_dynamic_api_exc(exc):
    if exc.body:
        if exc.headers and exc.headers.get('Content-Type') == 'application/json':
            message = json.loads(exc.body).get('message')
            if message:
                return message
        return exc.body
    else:
        return '%s Reason: %s' % (exc.status, exc.reason)