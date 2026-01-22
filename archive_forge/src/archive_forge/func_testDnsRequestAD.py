import DNS
import unittest
def testDnsRequestAD(self):
    dnsob = DNS.DnsRequest('example.org')
    ad_response = dnsob.req(qtype='A', timeout=1)
    self.assertTrue(ad_response.answers)
    self.assertEqual(ad_response.answers[0]['data'].count('.'), 3)
    self.assertEqual(ad_response.answers[0]['data'], '93.184.216.34')