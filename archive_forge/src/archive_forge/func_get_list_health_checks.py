from boto.route53 import exception
import random
import uuid
import xml.sax
import boto
from boto.connection import AWSAuthConnection
from boto import handler
import boto.jsonresponse
from boto.route53.record import ResourceRecordSets
from boto.route53.zone import Zone
from boto.compat import six, urllib
def get_list_health_checks(self, maxitems=None, marker=None):
    """
        Return a list of health checks

        :type maxitems: int
        :param maxitems: Maximum number of items to return

        :type marker: str
        :param marker: marker to get next set of items to list

        """
    params = {}
    if maxitems is not None:
        params['maxitems'] = maxitems
    if marker is not None:
        params['marker'] = marker
    uri = '/%s/healthcheck' % (self.Version,)
    response = self.make_request('GET', uri, params=params)
    body = response.read()
    boto.log.debug(body)
    if response.status >= 300:
        raise exception.DNSServerError(response.status, response.reason, body)
    e = boto.jsonresponse.Element(list_marker='HealthChecks', item_marker=('HealthCheck',))
    h = boto.jsonresponse.XmlHandler(e, None)
    h.parse(body)
    return e