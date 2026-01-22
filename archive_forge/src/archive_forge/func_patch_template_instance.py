from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
def patch_template_instance(self, resource, templateinstance):
    result = None
    try:
        result = resource.status.patch(templateinstance)
    except Exception as exc:
        self.fail_json(msg='Failed to migrate TemplateInstance {0} due to: {1}'.format(templateinstance['metadata']['name'], to_native(exc)))
    return result.to_dict()