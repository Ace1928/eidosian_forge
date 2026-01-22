from __future__ import (absolute_import, division, print_function)
import traceback
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
def parse_image_reference(self, image_ref):
    result, err = parse_docker_image_ref(image_ref, self.module)
    if result.get('digest'):
        self.fail_json(msg='Cannot import by ID, error with definition: %s' % image_ref)
    tag = result.get('tag') or None
    if not self.params.get('all') and (not tag):
        tag = 'latest'
    source = self.params.get('source')
    if not source:
        source = image_ref
    return dict(name=result.get('name'), tag=tag, source=image_ref)