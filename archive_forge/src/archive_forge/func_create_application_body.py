from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_application_body(self, template_name, template_body, smart_container=True):
    if not isinstance(smart_container, bool):
        error = 'expecting bool value for smart_container, got: %s' % smart_container
        return (None, error)
    body = {'name': self.app_name, 'svm': {'name': self.svm_name}, 'smart_container': smart_container, template_name: template_body}
    return (body, None)