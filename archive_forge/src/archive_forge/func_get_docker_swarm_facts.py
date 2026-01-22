from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.swarm import AnsibleDockerSwarmClient
from ansible_collections.community.docker.plugins.module_utils.common import RequestException
from ansible_collections.community.docker.plugins.module_utils.util import (
def get_docker_swarm_facts(self):
    try:
        return self.client.inspect_swarm()
    except APIError as exc:
        self.client.fail('Error inspecting docker swarm: %s' % to_native(exc))