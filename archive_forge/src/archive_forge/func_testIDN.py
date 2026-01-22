import DNS
import unittest
def testIDN(self):
    """Can we lookup an internationalized domain name?"""
    dnsobj = DNS.DnsRequest('xn--bb-eka.at')
    unidnsobj = DNS.DnsRequest('öbb.at')
    a_resp = dnsobj.qry(qtype='A', resulttype='text', timeout=1)
    ua_resp = unidnsobj.qry(qtype='A', resulttype='text', timeout=1)
    self.assertTrue(a_resp.answers)
    self.assertTrue(ua_resp.answers)
    self.assertEqual(ua_resp.answers[0]['data'], a_resp.answers[0]['data'])