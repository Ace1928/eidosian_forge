from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
def connect_containers(self):
    for name in self.parameters.connected:
        if not self.is_container_connected(name):
            if not self.check_mode:
                data = {'Container': name, 'EndpointConfig': None}
                self.client.post_json('/networks/{0}/connect', self.parameters.name, data=data)
            self.results['actions'].append('Connected container %s' % (name,))
            self.results['changed'] = True
            self.diff_tracker.add('connected.{0}'.format(name), parameter=True, active=False)