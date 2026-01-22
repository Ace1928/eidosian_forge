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
def ex_set_node_scheduling(self, node, on_host_maintenance=None, automatic_restart=None):
    """Set the maintenance behavior for the node.

        See `Scheduling <https://developers.google.com/compute/
        docs/instances#onhostmaintenance>`_ documentation for more info.

        :param  node: Node object
        :type   node: :class:`Node`

        :keyword  on_host_maintenance: Defines whether node should be
                                       terminated or migrated when host machine
                                       goes down. Acceptable values are:
                                       'MIGRATE' or 'TERMINATE' (If not
                                       supplied, value will be reset to GCE
                                       default value for the instance type.)
        :type     on_host_maintenance: ``str``

        :keyword  automatic_restart: Defines whether the instance should be
                                     automatically restarted when it is
                                     terminated by Compute Engine. (If not
                                     supplied, value will be set to the GCE
                                     default value for the instance type.)
        :type     automatic_restart: ``bool``

        :return:  True if successful.
        :rtype:   ``bool``
        """
    if not hasattr(node, 'name'):
        node = self.ex_get_node(node, 'all')
    if on_host_maintenance is not None:
        on_host_maintenance = on_host_maintenance.upper()
        ohm_values = ['MIGRATE', 'TERMINATE']
        if on_host_maintenance not in ohm_values:
            raise ValueError('on_host_maintenance must be one of %s' % ','.join(ohm_values))
    request = '/zones/{}/instances/{}/setScheduling'.format(node.extra['zone'].name, node.name)
    scheduling_data = {}
    if on_host_maintenance is not None:
        scheduling_data['onHostMaintenance'] = on_host_maintenance
    if automatic_restart is not None:
        scheduling_data['automaticRestart'] = automatic_restart
    self.connection.async_request(request, method='POST', data=scheduling_data)
    new_node = self.ex_get_node(node.name, node.extra['zone'])
    node.extra['scheduling'] = new_node.extra['scheduling']
    ohm = node.extra['scheduling'].get('onHostMaintenance')
    ar = node.extra['scheduling'].get('automaticRestart')
    success = True
    if on_host_maintenance not in [None, ohm]:
        success = False
    if automatic_restart not in [None, ar]:
        success = False
    return success