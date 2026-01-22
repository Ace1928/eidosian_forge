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
def ex_list_sslcertificates(self):
    """
        Retrieves the list of SslCertificate resources available to the
        specified project.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute
        * https://www.googleapis.com/auth/compute.readonly

        :return: A list of SSLCertificate objects.
        :rtype: ``list`` of :class:`GCESslCertificate`
        """
    list_data = []
    request = '/global/sslCertificates'
    response = self.connection.request(request, method='GET').object
    list_data = [self._to_sslcertificate(a) for a in response.get('items', [])]
    return list_data