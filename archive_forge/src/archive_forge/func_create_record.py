from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
def create_record(self, name, zone, type, data, extra=None):
    """
        Create a new record.

        :param name: Record name without the domain name (e.g. www).
                     Note: If you want to create a record for a base domain
                     name, you should specify empty string ('') for this
                     argument.
        :type  name: ``str``

        :param zone: Zone where the requested record is created.
        :type  zone: :class:`Zone`

        :param type: DNS record type (A, AAAA, ...).
        :type  type: :class:`RecordType`

        :param data: Data for the record (depends on the record type).
        :type  data: ``str``

        :param extra: Extra attributes (driver specific). (optional)
        :type extra: ``dict``

        :rtype: :class:`Record`
        """
    r_json = {'name': name, 'data': data, 'record_type': type}
    if extra is not None:
        r_json.update(extra)
    r_data = json.dumps({'zone_record': r_json})
    try:
        response = self.connection.request('/zones/%s/records' % zone.id, method='POST', data=r_data)
    except BaseHTTPError as e:
        raise PointDNSException(value=e.message, http_code=e.code, driver=self)
    record = self._to_record(response.object, zone=zone)
    return record