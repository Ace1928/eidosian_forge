from __future__ import (absolute_import, division, print_function)
import json
from time import sleep
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import AnsibleDockerClient
def get_all_nodes_inspect(self):
    """
        Returns Swarm node info as in 'docker node inspect' command about all registered nodes

        :return:
            Structure with information about all nodes
        """
    try:
        node_info = self.nodes()
    except APIError as exc:
        if exc.status_code == 503:
            self.fail('Cannot inspect node: To inspect node execute module on Swarm Manager')
        self.fail('Error while reading from Swarm manager: %s' % to_native(exc))
    except Exception as exc:
        self.fail('Error inspecting swarm node: %s' % exc)
    json_str = json.dumps(node_info, ensure_ascii=False)
    node_info = json.loads(json_str)
    return node_info