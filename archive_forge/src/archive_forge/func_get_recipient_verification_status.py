import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@requires(['RecipientTokenId'])
@api_action()
def get_recipient_verification_status(self, action, response, **kw):
    """
        Returns the recipient status.
        """
    return self.get_object(action, kw, response)