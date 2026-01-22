from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException, APIError
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import convert_filters
def get_docker_host_info(self):
    try:
        return self.client.info()
    except APIError as exc:
        self.client.fail('Error inspecting docker host: %s' % to_native(exc))