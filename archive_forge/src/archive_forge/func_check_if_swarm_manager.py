from __future__ import (absolute_import, division, print_function)
import json
from time import sleep
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import AnsibleDockerClient
def check_if_swarm_manager(self):
    """
        Checks if node role is set as Manager in Swarm. The node is the docker host on which module action
        is performed. The inspect_swarm() will fail if node is not a manager

        :return: True if node is Swarm Manager, False otherwise
        """
    try:
        self.inspect_swarm()
        return True
    except APIError:
        return False