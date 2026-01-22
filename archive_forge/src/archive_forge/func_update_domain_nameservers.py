import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.route53.domains import exceptions
def update_domain_nameservers(self, domain_name, nameservers):
    """
        This operation replaces the current set of name servers for
        the domain with the specified set of name servers. If you use
        Amazon Route 53 as your DNS service, specify the four name
        servers in the delegation set for the hosted zone for the
        domain.

        If successful, this operation returns an operation ID that you
        can use to track the progress and completion of the action. If
        the request is not completed successfully, the domain
        registrant will be notified by email.

        :type domain_name: string
        :param domain_name: The name of a domain.
        Type: String

        Default: None

        Constraints: The domain name can contain only the letters a through z,
            the numbers 0 through 9, and hyphen (-). Internationalized Domain
            Names are not supported.

        Required: Yes

        :type nameservers: list
        :param nameservers: A list of new name servers for the domain.
        Type: Complex

        Children: `Name`, `GlueIps`

        Required: Yes

        """
    params = {'DomainName': domain_name, 'Nameservers': nameservers}
    return self.make_request(action='UpdateDomainNameservers', body=json.dumps(params))