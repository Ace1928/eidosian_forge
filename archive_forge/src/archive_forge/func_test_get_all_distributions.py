from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.cloudfront import CloudFrontConnection
from boto.cloudfront.distribution import Distribution, DistributionConfig, DistributionSummary
from boto.cloudfront.origin import CustomOrigin
def test_get_all_distributions(self):
    body = b'\n        <DistributionList xmlns="http://cloudfront.amazonaws.com/doc/2010-11-01/">\n            <Marker></Marker>\n            <MaxItems>100</MaxItems>\n            <IsTruncated>false</IsTruncated>\n            <DistributionSummary>\n                <Id>EEEEEEEEEEEEE</Id>\n                <Status>InProgress</Status>\n                <LastModifiedTime>2014-02-03T11:03:41.087Z</LastModifiedTime>\n                <DomainName>abcdef12345678.cloudfront.net</DomainName>\n                <CustomOrigin>\n                    <DNSName>example.com</DNSName>\n                    <HTTPPort>80</HTTPPort>\n                    <HTTPSPort>443</HTTPSPort>\n                    <OriginProtocolPolicy>http-only</OriginProtocolPolicy>\n                </CustomOrigin>\n                <CNAME>static.example.com</CNAME>\n                <Enabled>true</Enabled>\n            </DistributionSummary>\n        </DistributionList>\n        '
    self.set_http_response(status_code=200, body=body)
    response = self.service_connection.get_all_distributions()
    self.assertTrue(isinstance(response, list))
    self.assertEqual(len(response), 1)
    self.assertTrue(isinstance(response[0], DistributionSummary))
    self.assertEqual(response[0].id, 'EEEEEEEEEEEEE')
    self.assertEqual(response[0].domain_name, 'abcdef12345678.cloudfront.net')
    self.assertEqual(response[0].status, 'InProgress')
    self.assertEqual(response[0].cnames, ['static.example.com'])
    self.assertEqual(response[0].enabled, True)
    self.assertTrue(isinstance(response[0].origin, CustomOrigin))
    self.assertEqual(response[0].origin.dns_name, 'example.com')
    self.assertEqual(response[0].origin.http_port, 80)
    self.assertEqual(response[0].origin.https_port, 443)
    self.assertEqual(response[0].origin.origin_protocol_policy, 'http-only')