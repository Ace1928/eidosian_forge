import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.route53.domains import exceptions
def update_domain_contact_privacy(self, domain_name, admin_privacy=None, registrant_privacy=None, tech_privacy=None):
    """
        This operation updates the specified domain contact's privacy
        setting. When the privacy option is enabled, personal
        information such as postal or email address is hidden from the
        results of a public WHOIS query. The privacy services are
        provided by the AWS registrar, Gandi. For more information,
        see the `Gandi privacy features`_.

        This operation only affects the privacy of the specified
        contact type (registrant, administrator, or tech). Successful
        acceptance returns an operation ID that you can use with
        GetOperationDetail to track the progress and completion of the
        action. If the request is not completed successfully, the
        domain registrant will be notified by email.

        :type domain_name: string
        :param domain_name: The name of a domain.
        Type: String

        Default: None

        Constraints: The domain name can contain only the letters a through z,
            the numbers 0 through 9, and hyphen (-). Internationalized Domain
            Names are not supported.

        Required: Yes

        :type admin_privacy: boolean
        :param admin_privacy: Whether you want to conceal contact information
            from WHOIS queries. If you specify true, WHOIS ("who is") queries
            will return contact information for our registrar partner, Gandi,
            instead of the contact information that you enter.
        Type: Boolean

        Default: None

        Valid values: `True` | `False`

        Required: No

        :type registrant_privacy: boolean
        :param registrant_privacy: Whether you want to conceal contact
            information from WHOIS queries. If you specify true, WHOIS ("who
            is") queries will return contact information for our registrar
            partner, Gandi, instead of the contact information that you enter.
        Type: Boolean

        Default: None

        Valid values: `True` | `False`

        Required: No

        :type tech_privacy: boolean
        :param tech_privacy: Whether you want to conceal contact information
            from WHOIS queries. If you specify true, WHOIS ("who is") queries
            will return contact information for our registrar partner, Gandi,
            instead of the contact information that you enter.
        Type: Boolean

        Default: None

        Valid values: `True` | `False`

        Required: No

        """
    params = {'DomainName': domain_name}
    if admin_privacy is not None:
        params['AdminPrivacy'] = admin_privacy
    if registrant_privacy is not None:
        params['RegistrantPrivacy'] = registrant_privacy
    if tech_privacy is not None:
        params['TechPrivacy'] = tech_privacy
    return self.make_request(action='UpdateDomainContactPrivacy', body=json.dumps(params))