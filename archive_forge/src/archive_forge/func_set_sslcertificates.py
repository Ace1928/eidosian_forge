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
def set_sslcertificates(self, sslcertificates):
    """
        Set the SSL Certificates for this TargetHTTPSProxy

        :param  sslcertificates: SSL Certificates to set.
        :type   sslcertificates: ``list`` of :class:`GCESslCertificate`

        :return:  True if successful
        :rtype:   ``bool``
        """
    return self.driver.ex_targethttpsproxy_set_sslcertificates(targethttpsproxy=self, sslcertificates=sslcertificates)