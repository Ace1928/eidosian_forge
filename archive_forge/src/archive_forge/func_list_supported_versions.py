import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def list_supported_versions(self):
    """
        Retrieve a list of all the identity versions which are supported by
        this installation.

        :rtype: ``list`` of :class:`.OpenStackIdentityVersion`
        """
    response = self.request('/', method='GET')
    result = self._to_versions(data=response.object['versions']['values'])
    result = sorted(result, key=lambda x: x.version)
    return result