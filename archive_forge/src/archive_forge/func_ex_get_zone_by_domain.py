import copy
import base64
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import ET, b, httplib
from libcloud.utils.xml import findall, findtext
from libcloud.utils.misc import get_new_obj, merge_valid_keys
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
def ex_get_zone_by_domain(self, domain):
    """
        Retrieve a zone object by the domain name.

        :param domain: The domain which should be used
        :type  domain: ``str``

        :rtype: :class:`Zone`
        """
    path = API_ROOT + 'zones/%s.xml' % domain
    self.connection.set_context({'resource': 'zone', 'id': domain})
    data = self.connection.request(path).object
    zone = self._to_zone(elem=data)
    return zone