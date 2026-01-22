from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.swarm import AnsibleDockerSwarmClient
from ansible_collections.community.docker.plugins.module_utils.common import RequestException
from ansible_collections.community.docker.plugins.module_utils.util import (
def get_docker_items_list(self, docker_object=None, filters=None):
    items = None
    items_list = []
    try:
        if docker_object == 'nodes':
            items = self.client.nodes(filters=filters)
        elif docker_object == 'tasks':
            items = self.client.tasks(filters=filters)
        elif docker_object == 'services':
            items = self.client.services(filters=filters)
    except APIError as exc:
        self.client.fail("Error inspecting docker swarm for object '%s': %s" % (docker_object, to_native(exc)))
    if self.verbose_output:
        return items
    for item in items:
        item_record = dict()
        if docker_object == 'nodes':
            item_record = self.get_essential_facts_nodes(item)
        elif docker_object == 'tasks':
            item_record = self.get_essential_facts_tasks(item)
        elif docker_object == 'services':
            item_record = self.get_essential_facts_services(item)
            if item_record['Mode'] == 'Global':
                item_record['Replicas'] = len(items)
        items_list.append(item_record)
    return items_list