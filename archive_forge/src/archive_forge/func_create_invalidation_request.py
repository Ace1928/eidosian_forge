import xml.sax
import time
import boto
from boto.connection import AWSAuthConnection
from boto import handler
from boto.cloudfront.distribution import Distribution, DistributionSummary, DistributionConfig
from boto.cloudfront.distribution import StreamingDistribution, StreamingDistributionSummary, StreamingDistributionConfig
from boto.cloudfront.identity import OriginAccessIdentity
from boto.cloudfront.identity import OriginAccessIdentitySummary
from boto.cloudfront.identity import OriginAccessIdentityConfig
from boto.cloudfront.invalidation import InvalidationBatch, InvalidationSummary, InvalidationListResultSet
from boto.resultset import ResultSet
from boto.cloudfront.exception import CloudFrontServerError
def create_invalidation_request(self, distribution_id, paths, caller_reference=None):
    """Creates a new invalidation request
            :see: http://goo.gl/8vECq
        """
    if not isinstance(paths, InvalidationBatch):
        paths = InvalidationBatch(paths)
    paths.connection = self
    uri = '/%s/distribution/%s/invalidation' % (self.Version, distribution_id)
    response = self.make_request('POST', uri, {'Content-Type': 'text/xml'}, data=paths.to_xml())
    body = response.read()
    if response.status == 201:
        h = handler.XmlHandler(paths, self)
        xml.sax.parseString(body, h)
        return paths
    else:
        raise CloudFrontServerError(response.status, response.reason, body)