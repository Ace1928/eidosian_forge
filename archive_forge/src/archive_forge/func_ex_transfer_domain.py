import re
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.types import LibcloudError
from libcloud.common.worldwidedns import WorldWideDNSConnection
def ex_transfer_domain(self, domain, user_id):
    """
        This command will allow you, if you are a reseller, to change the
        userid on a domain name to another userid in your account ONLY if that
        new userid is already created.

        :param domain: Domain name.
        :type  domain: ``str``

        :param user_id: The new userid to connect to the domain name.
        :type  user_id: ``str``

        :rtype: ``bool``

        For more info, please see:
        https://www.worldwidedns.net/dns_api_protocol_transfer.asp
        """
    if self.reseller_id is None:
        raise WorldWideDNSError('This is not a reseller account', driver=self)
    params = {'DOMAIN': domain, 'NEW_ID': user_id}
    response = self.connection.request('/api_dns_transfer.asp', params=params)
    return response.success()