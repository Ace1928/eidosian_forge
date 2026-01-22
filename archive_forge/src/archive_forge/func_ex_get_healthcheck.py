import hmac
import json
import base64
import datetime
from hashlib import sha256
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
def ex_get_healthcheck(self, zone, health_check_id):
    """
        Get a single Health Check from a zone

        :param zone: Zone in which the health check is
        :type zone: :class:`Zone`

        :param health_check_id: ID of the required health check
        :type  health_check_id: ``str``

        :return: :class:`AuroraDNSHealthCheck`
        """
    self.connection.set_context({'resource': 'healthcheck', 'id': health_check_id})
    res = self.connection.request('/zones/{}/health_checks/{}'.format(zone.id, health_check_id))
    check = res.parse_body()
    return self.__res_to_healthcheck(zone, check)