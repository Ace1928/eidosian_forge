from typing import Any, Dict, List, Optional, Tuple
from ansible_collections.kubernetes.core.plugins.module_utils.hashes import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.waiter import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible.module_utils.common.dict_transformations import dict_merge
def patch_resource(self, resource: Resource, definition: Dict, name: str, namespace: str, merge_type: str=None) -> Dict:
    if merge_type == 'json':
        self.module.deprecate(msg='json as a merge_type value is deprecated. Please use the k8s_json_patch module instead.', version='3.0.0', collection_name='kubernetes.core')
    try:
        params = dict(name=name, namespace=namespace)
        if merge_type:
            params['content_type'] = 'application/{0}-patch+json'.format(merge_type)
        return self.client.patch(resource, definition, **params).to_dict()
    except Exception as e:
        reason = e.body if hasattr(e, 'body') else e
        msg = 'Failed to patch object: {0}'.format(reason)
        raise CoreException(msg) from e