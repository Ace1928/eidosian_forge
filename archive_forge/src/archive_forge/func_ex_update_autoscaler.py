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
def ex_update_autoscaler(self, autoscaler):
    """
        Update an autoscaler with new values.

        To update, change the attributes of the autoscaler object and pass
        the updated object to the method.

        :param  autoscaler: An Autoscaler object with updated values.
        :type   autoscaler: :class:`GCEAutoscaler`

        :return:  An Autoscaler object representing the new state.
        :rtype:   :class:`GCEAutoscaler``
        """
    request = '/zones/%s/autoscalers' % autoscaler.zone.name
    as_data = {}
    as_data['name'] = autoscaler.name
    as_data['autoscalingPolicy'] = autoscaler.policy
    as_data['target'] = autoscaler.target.extra['selfLink']
    self.connection.async_request(request, method='PUT', data=as_data)
    return self.ex_get_autoscaler(autoscaler.name, autoscaler.zone)