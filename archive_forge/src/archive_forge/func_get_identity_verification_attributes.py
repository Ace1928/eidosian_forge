import re
import base64
from boto.compat import six, urllib
from boto.connection import AWSAuthConnection
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
import boto
import boto.jsonresponse
from boto.ses import exceptions as ses_exceptions
def get_identity_verification_attributes(self, identities):
    """Given a list of identities (email addresses and/or domains),
        returns the verification status and (for domain identities)
        the verification token for each identity.

        :type identities: list of strings or string
        :param identities: List of identities.

        :rtype: dict
        :returns: A GetIdentityVerificationAttributesResponse structure.
                  Note that keys must be unicode strings.
        """
    params = {}
    self._build_list_params(params, identities, 'Identities.member')
    return self._make_request('GetIdentityVerificationAttributes', params)