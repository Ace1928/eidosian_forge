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
@requires_some_of('ShipmentIdList', 'ShipmentStatusList')
@structured_lists('ShipmentIdList.Id', 'ShipmentStatusList.Status')
@api_action('Inbound', 30, 0.5)
def list_inbound_shipments(self, request, response, **kw):
    """Returns a list of inbound shipments based on criteria that
           you specify.
        """
    return self._post_request(request, kw, response)