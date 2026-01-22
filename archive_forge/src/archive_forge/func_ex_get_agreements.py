from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError
def ex_get_agreements(self, tld, privacy=True):
    """
        Get the legal agreements for a tld
        Use this in conjunction with ex_purchase_domain

        :param   tld: The top level domain e.g com, eu, uk
        :type    tld: ``str``

        :rtype: `dict` the JSON Schema
        """
    result = self.connection.request('/v1/domains/agreements', params={'tlds': tld, 'privacy': str(privacy)}, method='GET').object
    agreements = []
    for item in result:
        agreements.append(GoDaddyLegalAgreement(agreement_key=item['agreementKey'], title=item['title'], url=item['url'], content=item['content']))
    return agreements