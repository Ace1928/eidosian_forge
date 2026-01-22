from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError
from ansible_collections.kubernetes.core.plugins.module_utils.common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
def get_available_namespaces(self, client):
    v1_namespace = client.resources.get(api_version='v1', kind='Namespace')
    try:
        obj = v1_namespace.get()
    except DynamicApiError as exc:
        self.display.debug(exc)
        raise K8sInventoryException('Error fetching Namespace list: %s' % format_dynamic_api_exc(exc))
    return [namespace.metadata.name for namespace in obj.items]