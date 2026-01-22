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
def get_deployment_time(self):
    """
        Gets the date and time a vApp was deployed. Time is inferred from the
        deployment lease and expiration or the storage lease and expiration.

        :return: Date and time the vApp was deployed or None if unable to
                 calculate
        :rtype: ``datetime.datetime`` or ``None``
        """
    if self.deployment_lease is not None and self.deployment_lease_expiration is not None:
        return self.deployment_lease_expiration - datetime.timedelta(seconds=self.deployment_lease)
    if self.storage_lease is not None and self.storage_lease_expiration is not None:
        return self.storage_lease_expiration - datetime.timedelta(seconds=self.storage_lease)
    raise Exception('Cannot get time deployed. Missing complete lease and expiration information.')