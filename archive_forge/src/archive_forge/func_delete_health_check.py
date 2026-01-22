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
def delete_health_check(self, health_check_id):
    """
        Delete a health check

        :type health_check_id: str
        :param health_check_id: ID of the health check to delete

        """
    uri = '/%s/healthcheck/%s' % (self.Version, health_check_id)
    response = self.make_request('DELETE', uri)
    body = response.read()
    boto.log.debug(body)
    if response.status not in (200, 204):
        raise exception.DNSServerError(response.status, response.reason, body)
    e = boto.jsonresponse.Element()
    h = boto.jsonresponse.XmlHandler(e, None)
    h.parse(body)
    return e