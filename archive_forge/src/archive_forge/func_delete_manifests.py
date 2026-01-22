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
def delete_manifests(self, path, digests):
    for digest in digests:
        url = '%s/v2/%s/manifests/%s' % (self.registryhost, path, digest)
        self.changed = True
        if not self.check_mode:
            self.delete_from_registry(url=url)