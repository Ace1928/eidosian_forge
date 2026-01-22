import re
import base64
from boto.compat import six, urllib
from boto.connection import AWSAuthConnection
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
import boto
import boto.jsonresponse
from boto.ses import exceptions as ses_exceptions
def list_verified_email_addresses(self):
    """Fetch a list of the email addresses that have been verified.

        :rtype: dict
        :returns: A ListVerifiedEmailAddressesResponse structure. Note that
                  keys must be unicode strings.
        """
    return self._make_request('ListVerifiedEmailAddresses')