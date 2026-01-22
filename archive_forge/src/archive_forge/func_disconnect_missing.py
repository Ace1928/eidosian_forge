from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
def disconnect_missing(self):
    if not self.existing_network:
        return
    containers = self.existing_network['Containers']
    if not containers:
        return
    for c in containers.values():
        name = c['Name']
        if name not in self.parameters.connected:
            self.disconnect_container(name)