from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
from ansible_collections.kubernetes.core.plugins.module_utils.ansiblemodule import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.service import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.waiter import (
def json_patch(existing, patch):
    if not HAS_JSON_PATCH:
        error = {'msg': missing_required_lib('jsonpatch'), 'exception': JSON_PATCH_IMPORT_ERR}
        return (None, error)
    try:
        patch = jsonpatch.JsonPatch(patch)
        patched = patch.apply(existing)
        return (patched, None)
    except jsonpatch.InvalidJsonPatch as e:
        error = {'msg': 'Invalid JSON patch', 'exception': e}
        return (None, error)
    except jsonpatch.JsonPatchConflict as e:
        error = {'msg': 'Patch could not be applied due to a conflict', 'exception': e}
        return (None, error)