from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.cloudfront import CloudFrontConnection
from boto.cloudfront.distribution import Distribution, DistributionConfig, DistributionSummary
from boto.cloudfront.origin import CustomOrigin
def test_set_distribution_config(self):
    get_body = b'\n        <DistributionConfig xmlns="http://cloudfront.amazonaws.com/doc/2010-11-01/">\n        <CustomOrigin>\n            <DNSName>example.com</DNSName>\n            <HTTPPort>80</HTTPPort>\n            <HTTPSPort>443</HTTPSPort>\n            <OriginProtocolPolicy>http-only</OriginProtocolPolicy>\n        </CustomOrigin>\n        <CallerReference>1234567890123</CallerReference>\n        <CNAME>static.example.com</CNAME>\n        <Enabled>true</Enabled>\n        </DistributionConfig>\n        '
    put_body = b'\n        <Distribution xmlns="http://cloudfront.amazonaws.com/doc/2010-11-01/">\n            <Id>EEEEEE</Id>\n            <Status>InProgress</Status>\n            <LastModifiedTime>2014-02-04T10:47:53.493Z</LastModifiedTime>\n            <InProgressInvalidationBatches>0</InProgressInvalidationBatches>\n            <DomainName>d2000000000000.cloudfront.net</DomainName>\n            <DistributionConfig>\n                <CustomOrigin>\n                    <DNSName>example.com</DNSName>\n                    <HTTPPort>80</HTTPPort>\n                    <HTTPSPort>443</HTTPSPort>\n                    <OriginProtocolPolicy>match-viewer</OriginProtocolPolicy>\n                </CustomOrigin>\n                <CallerReference>aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee</CallerReference>\n                <Comment>this is a comment</Comment>\n                <Enabled>false</Enabled>\n            </DistributionConfig>\n        </Distribution>\n        '
    self.set_http_response(status_code=200, body=get_body, header={'Etag': 'AA'})
    conf = self.service_connection.get_distribution_config('EEEEEEE')
    self.set_http_response(status_code=200, body=put_body, header={'Etag': 'AABBCCD'})
    conf.comment = 'this is a comment'
    response = self.service_connection.set_distribution_config('EEEEEEE', conf.etag, conf)
    self.assertEqual(response, 'AABBCCD')