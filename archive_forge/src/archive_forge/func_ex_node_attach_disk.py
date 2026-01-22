from datetime import datetime
from libcloud.common.gandi import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_node_attach_disk(self, node, disk):
    """
        Specific method to attach a disk to a node

        :param      node: Node which should be used
        :type       node: :class:`Node`

        :param      disk: Disk which should be used
        :type       disk: :class:`GandiDisk`

        :rtype: ``bool``
        """
    op = self.connection.request('hosting.vm.disk_attach', int(node.id), int(disk.id))
    if self._wait_operation(op.object['id']):
        return True
    return False