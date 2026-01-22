from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
def ex_delete_redirect(self, redirect):
    """
        :param mail_r: Redirect to delete
        :type mail_r: :class:`Redirect`

        :rtype: ``bool``
        """
    zone_id = redirect.zone.id
    redirect_id = redirect.id
    try:
        self.connection.request('/zones/{}/redirects/{}'.format(zone_id, redirect_id), method='DELETE')
    except (BaseHTTPError, MalformedResponseError) as e:
        if isinstance(e, MalformedResponseError) and e.body == 'Not found':
            raise PointDNSException(value="Couldn't found redirect", http_code=httplib.NOT_FOUND, driver=self)
        raise PointDNSException(value=e.message, http_code=e.code, driver=self)
    return True