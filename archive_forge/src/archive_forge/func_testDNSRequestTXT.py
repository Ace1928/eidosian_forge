import DNS
import unittest
def testDNSRequestTXT(self):
    dnsobj = DNS.DnsRequest('fail.kitterman.org')
    respdef = dnsobj.qry(qtype='TXT', timeout=1)
    self.assertTrue(respdef.answers)
    data = respdef.answers[0]['data']
    self.assertEqual(data, [b'v=spf1 -all'])
    resptext = dnsobj.qry(qtype='TXT', resulttype='text', timeout=1)
    self.assertTrue(resptext.answers)
    data = resptext.answers[0]['data']
    self.assertEqual(data, ['v=spf1 -all'])
    respbin = dnsobj.qry(qtype='TXT', resulttype='binary', timeout=1)
    self.assertTrue(respbin.answers)
    data = respbin.answers[0]['data']
    self.assertEqual(data, [b'\x0bv=spf1 -all'])