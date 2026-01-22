import boto
import boto.jsonresponse
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def describe_domains(self, domain_names=None):
    """
        Describes the domains (optionally limited to one or more
        domains by name) owned by this account.

        :type domain_names: list
        :param domain_names: Limits the response to the specified domains.

        :raises: BaseException, InternalException
        """
    doc_path = ('describe_domains_response', 'describe_domains_result', 'domain_status_list')
    params = {}
    if domain_names:
        for i, domain_name in enumerate(domain_names, 1):
            params['DomainNames.member.%d' % i] = domain_name
    return self.get_response(doc_path, 'DescribeDomains', params, verb='POST', list_marker='DomainStatusList')