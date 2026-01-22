import DNS
import unittest
def testDnsRequestAAAAD(self):
    dnsob = DNS.DnsRequest('example.org')
    aaaad_response = dnsob.req(qtype='AAAA', timeout=1)
    self.assertTrue(aaaad_response.answers)
    self.assertEqual(len(aaaad_response.answers[0]['data']), 16)
    for b in aaaad_response.answers[0]['data']:
        assertIsByte(b)
    self.assertEqual(aaaad_response.answers[0]['data'], b'&\x06(\x00\x02 \x00\x01\x02H\x18\x93%\xc8\x19F')