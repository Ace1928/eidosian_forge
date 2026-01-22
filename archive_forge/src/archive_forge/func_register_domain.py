import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.route53.domains import exceptions
def register_domain(self, domain_name, duration_in_years, admin_contact, registrant_contact, tech_contact, idn_lang_code=None, auto_renew=None, privacy_protect_admin_contact=None, privacy_protect_registrant_contact=None, privacy_protect_tech_contact=None):
    """
        This operation registers a domain. Domains are registered by
        the AWS registrar partner, Gandi. For some top-level domains
        (TLDs), this operation requires extra parameters.

        When you register a domain, Amazon Route 53 does the
        following:


        + Creates a Amazon Route 53 hosted zone that has the same name
          as the domain. Amazon Route 53 assigns four name servers to
          your hosted zone and automatically updates your domain
          registration with the names of these name servers.
        + Enables autorenew, so your domain registration will renew
          automatically each year. We'll notify you in advance of the
          renewal date so you can choose whether to renew the
          registration.
        + Optionally enables privacy protection, so WHOIS queries
          return contact information for our registrar partner, Gandi,
          instead of the information you entered for registrant, admin,
          and tech contacts.
        + If registration is successful, returns an operation ID that
          you can use to track the progress and completion of the
          action. If the request is not completed successfully, the
          domain registrant is notified by email.
        + Charges your AWS account an amount based on the top-level
          domain. For more information, see `Amazon Route 53 Pricing`_.

        :type domain_name: string
        :param domain_name: The name of a domain.
        Type: String

        Default: None

        Constraints: The domain name can contain only the letters a through z,
            the numbers 0 through 9, and hyphen (-). Internationalized Domain
            Names are not supported.

        Required: Yes

        :type idn_lang_code: string
        :param idn_lang_code: Reserved for future use.

        :type duration_in_years: integer
        :param duration_in_years: The number of years the domain will be
            registered. Domains are registered for a minimum of one year. The
            maximum period depends on the top-level domain.
        Type: Integer

        Default: 1

        Valid values: Integer from 1 to 10

        Required: Yes

        :type auto_renew: boolean
        :param auto_renew: Indicates whether the domain will be automatically
            renewed ( `True`) or not ( `False`). Autorenewal only takes effect
            after the account is charged.
        Type: Boolean

        Valid values: `True` | `False`

        Default: `True`

        Required: No

        :type admin_contact: dict
        :param admin_contact: Provides detailed contact information.
        Type: Complex

        Children: `FirstName`, `MiddleName`, `LastName`, `ContactType`,
            `OrganizationName`, `AddressLine1`, `AddressLine2`, `City`,
            `State`, `CountryCode`, `ZipCode`, `PhoneNumber`, `Email`, `Fax`,
            `ExtraParams`

        Required: Yes

        :type registrant_contact: dict
        :param registrant_contact: Provides detailed contact information.
        Type: Complex

        Children: `FirstName`, `MiddleName`, `LastName`, `ContactType`,
            `OrganizationName`, `AddressLine1`, `AddressLine2`, `City`,
            `State`, `CountryCode`, `ZipCode`, `PhoneNumber`, `Email`, `Fax`,
            `ExtraParams`

        Required: Yes

        :type tech_contact: dict
        :param tech_contact: Provides detailed contact information.
        Type: Complex

        Children: `FirstName`, `MiddleName`, `LastName`, `ContactType`,
            `OrganizationName`, `AddressLine1`, `AddressLine2`, `City`,
            `State`, `CountryCode`, `ZipCode`, `PhoneNumber`, `Email`, `Fax`,
            `ExtraParams`

        Required: Yes

        :type privacy_protect_admin_contact: boolean
        :param privacy_protect_admin_contact: Whether you want to conceal
            contact information from WHOIS queries. If you specify true, WHOIS
            ("who is") queries will return contact information for our
            registrar partner, Gandi, instead of the contact information that
            you enter.
        Type: Boolean

        Default: `True`

        Valid values: `True` | `False`

        Required: No

        :type privacy_protect_registrant_contact: boolean
        :param privacy_protect_registrant_contact: Whether you want to conceal
            contact information from WHOIS queries. If you specify true, WHOIS
            ("who is") queries will return contact information for our
            registrar partner, Gandi, instead of the contact information that
            you enter.
        Type: Boolean

        Default: `True`

        Valid values: `True` | `False`

        Required: No

        :type privacy_protect_tech_contact: boolean
        :param privacy_protect_tech_contact: Whether you want to conceal
            contact information from WHOIS queries. If you specify true, WHOIS
            ("who is") queries will return contact information for our
            registrar partner, Gandi, instead of the contact information that
            you enter.
        Type: Boolean

        Default: `True`

        Valid values: `True` | `False`

        Required: No

        """
    params = {'DomainName': domain_name, 'DurationInYears': duration_in_years, 'AdminContact': admin_contact, 'RegistrantContact': registrant_contact, 'TechContact': tech_contact}
    if idn_lang_code is not None:
        params['IdnLangCode'] = idn_lang_code
    if auto_renew is not None:
        params['AutoRenew'] = auto_renew
    if privacy_protect_admin_contact is not None:
        params['PrivacyProtectAdminContact'] = privacy_protect_admin_contact
    if privacy_protect_registrant_contact is not None:
        params['PrivacyProtectRegistrantContact'] = privacy_protect_registrant_contact
    if privacy_protect_tech_contact is not None:
        params['PrivacyProtectTechContact'] = privacy_protect_tech_contact
    return self.make_request(action='RegisterDomain', body=json.dumps(params))