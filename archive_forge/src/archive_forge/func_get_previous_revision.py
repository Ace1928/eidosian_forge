from __future__ import absolute_import, division, print_function
import copy
from ansible_collections.kubernetes.core.plugins.module_utils.ansiblemodule import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.service import (
def get_previous_revision(all_resources, current_revision):
    for resource in all_resources:
        if resource['kind'] == 'ReplicaSet':
            if int(resource['metadata']['annotations']['deployment.kubernetes.io/revision']) == int(current_revision) - 1:
                return resource
        elif resource['kind'] == 'ControllerRevision':
            if int(resource['metadata']['annotations']['deprecated.daemonset.template.generation']) == int(current_revision) - 1:
                return resource
    return None