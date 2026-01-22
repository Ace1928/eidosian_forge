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
def create_health_check(self, health_check, caller_ref=None):
    """
        Create a new Health Check

        :type health_check: HealthCheck
        :param health_check: HealthCheck object

        :type caller_ref: str
        :param caller_ref: A unique string that identifies the request
            and that allows failed CreateHealthCheckRequest requests to be retried
            without the risk of executing the operation twice.  If you don't
            provide a value for this, boto will generate a Type 4 UUID and
            use that.

        """
    if caller_ref is None:
        caller_ref = str(uuid.uuid4())
    uri = '/%s/healthcheck' % self.Version
    params = {'xmlns': self.XMLNameSpace, 'caller_ref': caller_ref, 'health_check': health_check.to_xml()}
    xml_body = self.POSTHCXMLBody % params
    response = self.make_request('POST', uri, {'Content-Type': 'text/xml'}, xml_body)
    body = response.read()
    boto.log.debug(body)
    if response.status == 201:
        e = boto.jsonresponse.Element()
        h = boto.jsonresponse.XmlHandler(e, None)
        h.parse(body)
        return e
    else:
        raise exception.DNSServerError(response.status, response.reason, body)