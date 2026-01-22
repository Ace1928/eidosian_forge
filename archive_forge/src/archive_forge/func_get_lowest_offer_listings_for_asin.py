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
@requires(['MarketplaceId', 'ASINList'])
@structured_lists('ASINList.ASIN')
@api_action('Products', 20, 5, 'GetLowestOfferListingsForASIN')
def get_lowest_offer_listings_for_asin(self, request, response, **kw):
    """Returns the lowest price offer listings for a specific
           product by item condition and ASINs.
        """
    return self._post_request(request, kw, response)