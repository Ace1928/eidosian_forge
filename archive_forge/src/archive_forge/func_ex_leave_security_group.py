import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
def ex_leave_security_group(self, node, group_id=None):
    """
        Leave a node from security group.

        :param node: The node to leave security group
        :type node: :class:`Node`

        :param group_id: security group id.
        :type group_id: ``str``


        :return: leave operation result.
        :rtype: ``bool``
        """
    if group_id is None:
        raise AttributeError('group_id is required')
    if node.state != NodeState.RUNNING and node.state != NodeState.STOPPED:
        raise LibcloudError('The node state with id % s need                                be running or stopped .' % node.id)
    params = {'Action': 'LeaveSecurityGroup', 'InstanceId': node.id, 'SecurityGroupId': group_id}
    resp = self.connection.request(self.path, params)
    return resp.success()