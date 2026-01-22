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
@requires(['MarketplaceId', 'NotificationType', 'Destination'])
@structured_objects('Destination', members=True)
@api_action('Subscriptions', 25, 0.5)
def delete_subscription(self, request, response, **kw):
    """Deletes the subscription for the specified notification type and
           destination.
        """
    return self._post_request(request, kw, response)