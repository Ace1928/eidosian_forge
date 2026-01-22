from __future__ import (absolute_import, division, print_function)
import traceback
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
def create_image_stream_import(self, stream):
    isi = {'apiVersion': 'image.openshift.io/v1', 'kind': 'ImageStreamImport', 'metadata': {'name': stream['metadata']['name'], 'namespace': stream['metadata']['namespace'], 'resourceVersion': stream['metadata'].get('resourceVersion')}, 'spec': {'import': True}}
    annotations = stream.get('annotations', {})
    insecure = boolean(annotations.get('openshift.io/image.insecureRepository', True))
    if self.validate_certs is not None:
        insecure = not self.validate_certs
    return (isi, insecure)