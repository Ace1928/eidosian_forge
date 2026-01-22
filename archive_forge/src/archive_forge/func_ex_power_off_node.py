import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_power_off_node(self, node):
    """
        Powers on all VMs under specified node. VMs need to be This operation
        is allowed only when the vApp/VM is powered on.

        :param  node: The node to be powered off
        :type   node: :class:`Node`

        :rtype: :class:`Node`
        """
    return self._perform_power_operation(node, 'powerOff')