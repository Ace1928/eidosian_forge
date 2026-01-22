import re
import base64
from boto.compat import six, urllib
from boto.connection import AWSAuthConnection
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
import boto
import boto.jsonresponse
from boto.ses import exceptions as ses_exceptions
def delete_identity(self, identity):
    """Deletes the specified identity (email address or domain) from
        the list of verified identities.

        :type identity: string
        :param identity: The identity to be deleted.

        :rtype: dict
        :returns: A DeleteIdentityResponse structure. Note that keys must
                  be unicode strings.
        """
    return self._make_request('DeleteIdentity', {'Identity': identity})