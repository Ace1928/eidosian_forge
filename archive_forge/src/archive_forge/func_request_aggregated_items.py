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
def request_aggregated_items(self, api_name, zone=None):
    """
        Perform request(s) to obtain all results from 'api_name'.

        This method will make requests to the aggregated 'api_name' until
        all results are received.  It will then, through a helper function,
        combine all results and return a single 'items' dictionary.

        :param    api_name: Name of API to call. Consult API docs
                  for valid names.
        :type     api_name: ``str``

        :param   zone: Optional zone to use.
        :type zone: :class:`GCEZone`

        :return:  dict in the format of the API response.
                  format: { 'items': {'key': {api_name: []}} }
                  ex: { 'items': {'zones/us-central1-a': {disks: []}} }
        :rtype:   ``dict``
        """
    if zone:
        request_path = '/zones/{}/{}'.format(zone.name, api_name)
    else:
        request_path = '/aggregated/%s' % api_name
    api_responses = []
    params = {'maxResults': 500}
    more_results = True
    while more_results:
        self.gce_params = params
        response = self.request(request_path, method='GET').object
        if 'items' in response:
            if zone:
                items = response['items']
                response['items'] = {'zones/%s' % zone: {api_name: items}}
            api_responses.append(response)
        more_results = 'pageToken' in params
    return self._merge_response_items(api_name, api_responses)