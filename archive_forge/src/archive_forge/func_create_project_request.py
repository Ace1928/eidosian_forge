from __future__ import (absolute_import, division, print_function)
import re
import operator
from functools import reduce
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible.module_utils._text import to_native
def create_project_request(self, definition):
    definition['kind'] = 'ProjectRequest'
    result = {'changed': False, 'result': {}}
    resource = self.svc.find_resource(kind='ProjectRequest', api_version=definition['apiVersion'], fail=True)
    if not self.check_mode:
        try:
            k8s_obj = resource.create(definition)
            result['result'] = k8s_obj.to_dict()
        except DynamicApiError as exc:
            self.fail_json(msg='Failed to create object: {0}'.format(exc.body), error=exc.status, status=exc.status, reason=exc.reason)
    result['changed'] = True
    result['method'] = 'create'
    return result