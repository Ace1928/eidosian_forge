import base64
from libcloud.utils.py3 import b
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
Create a new VPS.net node

        @inherits: :class:`NodeDriver.create_node`

        :keyword    ex_backups_enabled: Enable automatic backups
        :type       ex_backups_enabled: ``bool``

        :keyword    ex_fqdn:   Fully Qualified domain of the node
        :type       ex_fqdn:   ``str``
        