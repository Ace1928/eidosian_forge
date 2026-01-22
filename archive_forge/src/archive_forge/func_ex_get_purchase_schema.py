from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError
def ex_get_purchase_schema(self, tld):
    """
        Get the schema that needs completing to purchase a new domain
        Use this in conjunction with ex_purchase_domain

        :param   tld: The top level domain e.g com, eu, uk
        :type    tld: ``str``

        :rtype: `dict` the JSON Schema
        """
    result = self.connection.request('/v1/domains/purchase/schema/%s' % tld, method='GET').object
    return result