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
@requires(['MarketplaceId'])
@structured_lists('CategoryQueryList.CategoryQuery')
@api_action('Recommendations', 5, 2)
def list_recommendations(self, request, response, **kw):
    """Returns your active recommendations for a specific category or for
           all categories for a specific marketplace.
        """
    return self._post_request(request, kw, response)