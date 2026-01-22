import DNS
import unittest
def testDNSRequestTXTD(self):
    dnsob = DNS.DnsRequest('fail.kitterman.org')
    respdef = dnsob.req(qtype='TXT', timeout=1)
    self.assertTrue(respdef.answers)
    data = respdef.answers[0]['data']
    self.assertEqual(data, [b'v=spf1 -all'])