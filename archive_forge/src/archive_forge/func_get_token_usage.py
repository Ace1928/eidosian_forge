import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@requires(['TokenId'])
@api_action()
def get_token_usage(self, action, response, **kw):
    """
        Returns the usage of a token.
        """
    return self.get_object(action, kw, response)