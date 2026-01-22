from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.swarm import AnsibleDockerSwarmClient
from ansible.module_utils.common.text.converters import to_native
def compare_to_active(self, other, client, differences):
    for k in self.__dict__:
        if k in ('advertise_addr', 'listen_addr', 'remote_addrs', 'join_token', 'rotate_worker_token', 'rotate_manager_token', 'spec', 'default_addr_pool', 'subnet_size', 'data_path_addr', 'data_path_port'):
            continue
        if not client.option_minimal_versions[k]['supported']:
            continue
        value = getattr(self, k)
        if value is None:
            continue
        other_value = getattr(other, k)
        if value != other_value:
            differences.add(k, parameter=value, active=other_value)
    if self.rotate_worker_token:
        differences.add('rotate_worker_token', parameter=True, active=False)
    if self.rotate_manager_token:
        differences.add('rotate_manager_token', parameter=True, active=False)
    return differences