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
def delete_or_evict_pods(self, node_unschedulable):
    result = []
    if not node_unschedulable:
        self.patch_node(unschedulable=True)
        result.append('node {0} marked unschedulable.'.format(self._module.params.get('name')))
        self._changed = True
    else:
        result.append('node {0} already marked unschedulable.'.format(self._module.params.get('name')))

    def _revert_node_patch():
        if self._changed:
            self._changed = False
            self.patch_node(unschedulable=False)
    try:
        field_selector = 'spec.nodeName={name}'.format(name=self._module.params.get('name'))
        pod_list = self._api_instance.list_pod_for_all_namespaces(field_selector=field_selector)
        force = self._drain_options.get('force', False)
        ignore_daemonset = self._drain_options.get('ignore_daemonsets', False)
        delete_emptydir_data = self._drain_options.get('delete_emptydir_data', False)
        pods, warnings, errors = filter_pods(pod_list.items, force, ignore_daemonset, delete_emptydir_data)
        if errors:
            _revert_node_patch()
            self._module.fail_json(msg='Pod deletion errors: {0}'.format(' '.join(errors)))
    except ApiException as exc:
        if exc.reason != 'Not Found':
            _revert_node_patch()
            self._module.fail_json(msg='Failed to list pod from node {name} due to: {reason}'.format(name=self._module.params.get('name'), reason=exc.reason), status=exc.status)
        pods = []
    except Exception as exc:
        _revert_node_patch()
        self._module.fail_json(msg='Failed to list pod from node {name} due to: {error}'.format(name=self._module.params.get('name'), error=to_native(exc)))
    if pods:
        self.evict_pods(pods)
        number_pod = len(pods)
        if self._drain_options.get('wait_timeout') is not None:
            warn = self.wait_for_pod_deletion(pods, self._drain_options.get('wait_timeout'), self._drain_options.get('wait_sleep'))
            if warn:
                warnings.append(warn)
        result.append('{0} Pod(s) deleted from node.'.format(number_pod))
    if warnings:
        return dict(result=' '.join(result), warnings=warnings)
    return dict(result=' '.join(result))