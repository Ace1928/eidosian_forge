import DNS
import unittest
def testDnsRequestMX(self):
    dnsobj = DNS.DnsRequest('ietf.org')
    mx_response = dnsobj.qry(qtype='MX', timeout=1)
    self.assertTrue(mx_response.answers[0])
    self.assertEqual(mx_response.answers[0]['data'], (0, 'mail.ietf.org'))
    m = DNS.mxlookup('ietf.org', timeout=1)
    self.assertEqual(mx_response.answers[0]['data'], m[0])