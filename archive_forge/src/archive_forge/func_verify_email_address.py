import re
import base64
from boto.compat import six, urllib
from boto.connection import AWSAuthConnection
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
import boto
import boto.jsonresponse
from boto.ses import exceptions as ses_exceptions
def verify_email_address(self, email_address):
    """Verifies an email address. This action causes a confirmation email
        message to be sent to the specified address.

        :type email_adddress: string
        :param email_address: The email address to be verified.

        :rtype: dict
        :returns: A VerifyEmailAddressResponse structure. Note that keys must
                  be unicode strings.
        """
    return self._make_request('VerifyEmailAddress', {'EmailAddress': email_address})