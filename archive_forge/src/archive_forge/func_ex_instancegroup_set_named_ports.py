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
def ex_instancegroup_set_named_ports(self, instancegroup, named_ports=[]):
    """
        Sets the named ports for the specified instance group.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  instancegroup:  The Instance Group where where the
                                named ports are updated.
        :type   instancegroup: :class:`GCEInstanceGroup`

        :param  named_ports:  Assigns a name to a port number. For example:
                              {name: "http", port: 80}  This allows the
                              system to reference ports by the assigned name
                              instead of a port number. Named ports can also
                              contain multiple ports. For example: [{name:
                              "http", port: 80},{name: "http", port: 8080}]
                              Named ports apply to all instances in this
                              instance group.
        :type   named_ports: ``list`` of {'name': ``str``, 'port`: ``int``}

        :return:  Return True if successful.
        :rtype: ``bool``
        """
    if not isinstance(named_ports, list):
        raise ValueError("'named_ports' must be a list of name/port dictionaries.")
    request = '/zones/{}/instanceGroups/{}/setNamedPorts'.format(instancegroup.zone.name, instancegroup.name)
    request_data = {'namedPorts': named_ports, 'fingerprint': instancegroup.extra['fingerprint']}
    self.connection.async_request(request, method='POST', data=request_data)
    return True