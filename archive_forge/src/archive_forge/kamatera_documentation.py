import json
import time
import datetime
from libcloud.utils.py3 import basestring
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeDriver, NodeLocation
from libcloud.compute.types import Provider

        Get a Kamatera node object.

        :param id:   Node ID (optional)
        :type id:   ``str``

        :param name:   Node name (optional)
        :type name:   ``str``

        :param state:   Node state (optional)
        :type state:   :class:`libcloud.compute.types.NodeState`

        :param public_ips:   Node public IPS. (optional)
        :type public_ips:   ``list`` of :str:

        :param private_ips:   Node private IPS. (optional)
        :type private_ips:   ``list`` of :str:

        :param size:  node size. (optional)
        :type size:  :class:`.NodeSize`

        :param image:  Node OS Image. (optional)
        :type image:  :class:`.NodeImage`

        :param created_at:  Node creation time. (optional)
        :type created_at:  ``datetime.datetime``

        :param location: Node datacenter. (optional)
        :type location: :class:`.NodeLocation`

        :param dailybackup:   create daily backups for the node (optional)
        :type dailybackup:    ``bool``

        :param managed:   provide managed support for the node (optional)
        :type managed:    ``bool``

        :param billingcycle:   billing cycle (hourly / monthly) (optional)
        :type billingcycle:    ``str``

        :param generated_password:   server generated password (optional)
        :type generated_password:    ``str``

        :param create_command_id:   creation task command ID (optional)
        :type create_command_id:    ``int``

        :param poweronaftercreate:   power on the node after create (optional)
        :type poweronaftercreate:    ``bool``

        :return: The node.
        :rtype: :class:`.Node`
        