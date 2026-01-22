from boto.connection import AWSQueryConnection
from boto.provider import Provider, NO_CREDENTIALS_PROVIDED
from boto.regioninfo import RegionInfo
from boto.sts.credentials import Credentials, FederationToken, AssumedRole
from boto.sts.credentials import DecodeAuthorizationMessage
import boto
import boto.utils
import datetime
import threading
def decode_authorization_message(self, encoded_message):
    """
        Decodes additional information about the authorization status
        of a request from an encoded message returned in response to
        an AWS request.

        For example, if a user is not authorized to perform an action
        that he or she has requested, the request returns a
        `Client.UnauthorizedOperation` response (an HTTP 403
        response). Some AWS actions additionally return an encoded
        message that can provide details about this authorization
        failure.
        Only certain AWS actions return an encoded authorization
        message. The documentation for an individual action indicates
        whether that action returns an encoded message in addition to
        returning an HTTP code.
        The message is encoded because the details of the
        authorization status can constitute privileged information
        that the user who requested the action should not see. To
        decode an authorization status message, a user must be granted
        permissions via an IAM policy to request the
        `DecodeAuthorizationMessage` (
        `sts:DecodeAuthorizationMessage`) action.

        The decoded message includes the following type of
        information:


        + Whether the request was denied due to an explicit deny or
          due to the absence of an explicit allow. For more information,
          see `Determining Whether a Request is Allowed or Denied`_ in
          Using IAM .
        + The principal who made the request.
        + The requested action.
        + The requested resource.
        + The values of condition keys in the context of the user's
          request.

        :type encoded_message: string
        :param encoded_message: The encoded message that was returned with the
            response.

        """
    params = {'EncodedMessage': encoded_message}
    return self.get_object('DecodeAuthorizationMessage', params, DecodeAuthorizationMessage, verb='POST')