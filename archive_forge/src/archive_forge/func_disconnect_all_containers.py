from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
def disconnect_all_containers(self):
    containers = self.client.get_network(name=self.parameters.name)['Containers']
    if not containers:
        return
    for cont in containers.values():
        self.disconnect_container(cont['Name'])