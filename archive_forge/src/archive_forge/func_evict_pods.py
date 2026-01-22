from __future__ import absolute_import, division, print_function
import copy
import time
import traceback
from datetime import datetime
from ansible_collections.kubernetes.core.plugins.module_utils.ansiblemodule import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible.module_utils._text import to_native
def evict_pods(self, pods):
    for namespace, name in pods:
        try:
            if self._drain_options.get('disable_eviction'):
                self._api_instance.delete_namespaced_pod(name=name, namespace=namespace, body=self._delete_options)
            else:
                body = v1_eviction(delete_options=self._delete_options, metadata=V1ObjectMeta(name=name, namespace=namespace))
                self._api_instance.create_namespaced_pod_eviction(name=name, namespace=namespace, body=body)
            self._changed = True
        except ApiException as exc:
            if exc.reason != 'Not Found':
                self._module.fail_json(msg='Failed to delete pod {0}/{1} due to: {2}'.format(namespace, name, exc.reason))
        except Exception as exc:
            self._module.fail_json(msg='Failed to delete pod {0}/{1} due to: {2}'.format(namespace, name, to_native(exc)))