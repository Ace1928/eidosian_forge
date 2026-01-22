from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.community.okd.plugins.module_utils.openshift_images_common import (
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
def exceeds_limits(self, namespace, image):
    if namespace not in self.limit_range:
        return False
    docker_image_metadata = image.get('dockerImageMetadata')
    if not docker_image_metadata:
        return False
    docker_image_size = docker_image_metadata['Size']
    for limit in self.limit_range.get(namespace):
        for item in limit['spec']['limits']:
            if item['type'] != 'openshift.io/Image':
                continue
            limit_max = item['max']
            if not limit_max:
                continue
            storage = limit_max['storage']
            if not storage:
                continue
            if convert_storage_to_bytes(storage) < docker_image_size:
                return True
    return False