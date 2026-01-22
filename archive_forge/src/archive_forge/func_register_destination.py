from collections import abc
import xml.sax
import hashlib
import string
from boto.connection import AWSQueryConnection
from boto.exception import BotoServerError
import boto.mws.exception
import boto.mws.response
from boto.handler import XmlHandler
from boto.compat import filter, map, six, encodebytes
@requires(['MarketplaceId', 'Destination'])
@structured_objects('Destination', members=True)
@api_action('Subscriptions', 25, 0.5)
def register_destination(self, request, response, **kw):
    """Specifies a new destination where you want to receive notifications.
        """
    return self._post_request(request, kw, response)