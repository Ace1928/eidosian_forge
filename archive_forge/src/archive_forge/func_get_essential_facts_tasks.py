from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.swarm import AnsibleDockerSwarmClient
from ansible_collections.community.docker.plugins.module_utils.common import RequestException
from ansible_collections.community.docker.plugins.module_utils.util import (
def get_essential_facts_tasks(self, item):
    object_essentials = dict()
    object_essentials['ID'] = item['ID']
    object_essentials['ContainerID'] = item['Status']['ContainerStatus']['ContainerID']
    object_essentials['Image'] = item['Spec']['ContainerSpec']['Image']
    object_essentials['Node'] = self.client.get_node_name_by_id(item['NodeID'])
    object_essentials['DesiredState'] = item['DesiredState']
    object_essentials['CurrentState'] = item['Status']['State']
    if 'Err' in item['Status']:
        object_essentials['Error'] = item['Status']['Err']
    else:
        object_essentials['Error'] = None
    return object_essentials