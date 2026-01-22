import re
import base64
from boto.compat import six, urllib
from boto.connection import AWSAuthConnection
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
import boto
import boto.jsonresponse
from boto.ses import exceptions as ses_exceptions
def verify_domain_identity(self, domain):
    """Verifies a domain.

        :type domain: string
        :param domain: The domain to be verified.

        :rtype: dict
        :returns: A VerifyDomainIdentityResponse structure. Note that
                  keys must be unicode strings.
        """
    return self._make_request('VerifyDomainIdentity', {'Domain': domain})