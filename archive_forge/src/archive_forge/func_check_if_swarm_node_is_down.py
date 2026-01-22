from __future__ import (absolute_import, division, print_function)
import json
from time import sleep
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import AnsibleDockerClient
def check_if_swarm_node_is_down(self, node_id=None, repeat_check=1):
    """
        Checks if node status on Swarm manager is 'down'. If node_id is provided it query manager about
        node specified in parameter, otherwise it query manager itself. If run on Swarm Worker node or
        host that is not part of Swarm it will fail the playbook

        :param repeat_check: number of check attempts with 5 seconds delay between them, by default check only once
        :param node_id: node ID or name, if None then method will try to get node_id of host module run on
        :return:
            True if node is part of swarm but its state is down, False otherwise
        """
    if repeat_check < 1:
        repeat_check = 1
    if node_id is None:
        node_id = self.get_swarm_node_id()
    for retry in range(0, repeat_check):
        if retry > 0:
            sleep(5)
        node_info = self.get_node_inspect(node_id=node_id)
        if node_info['Status']['State'] == 'down':
            return True
    return False