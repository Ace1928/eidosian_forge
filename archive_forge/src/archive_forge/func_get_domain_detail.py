import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.route53.domains import exceptions
def get_domain_detail(self, domain_name):
    """
        This operation returns detailed information about the domain.
        The domain's contact information is also returned as part of
        the output.

        :type domain_name: string
        :param domain_name: The name of a domain.
        Type: String

        Default: None

        Constraints: The domain name can contain only the letters a through z,
            the numbers 0 through 9, and hyphen (-). Internationalized Domain
            Names are not supported.

        Required: Yes

        """
    params = {'DomainName': domain_name}
    return self.make_request(action='GetDomainDetail', body=json.dumps(params))