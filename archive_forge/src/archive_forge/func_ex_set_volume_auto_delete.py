import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_set_volume_auto_delete(self, volume, node, auto_delete=True):
    """
        Sets the auto-delete flag for a volume attached to a node.

        :param  volume: Volume object to auto-delete
        :type   volume: :class:`StorageVolume`

        :param   ex_node: Node object to auto-delete volume from
        :type    ex_node: :class:`Node`

        :keyword auto_delete: Flag to set for the auto-delete value
        :type    auto_delete: ``bool`` (default True)

        :return:  True if successful
        :rtype:   ``bool``
        """
    request = '/zones/{}/instances/{}/setDiskAutoDelete'.format(node.extra['zone'].name, node.name)
    delete_params = {'deviceName': volume.name, 'autoDelete': auto_delete}
    self.connection.async_request(request, method='POST', params=delete_params)
    return True