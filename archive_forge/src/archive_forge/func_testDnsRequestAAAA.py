import DNS
import unittest
def testDnsRequestAAAA(self):
    dnsobj = DNS.DnsRequest('example.org')
    aaaa_response = dnsobj.qry(qtype='AAAA', resulttype='text', timeout=1)
    self.assertTrue(aaaa_response.answers)
    self.assertTrue(':' in aaaa_response.answers[0]['data'])
    self.assertEqual(aaaa_response.answers[0]['data'], '2606:2800:220:1:248:1893:25c8:1946')
    aaaad_response = dnsobj.qry(qtype='AAAA', timeout=1)
    self.assertTrue(aaaad_response.answers)
    self.assertEqual(aaaad_response.answers[0]['data'], ipaddress.IPv6Address('2606:2800:220:1:248:1893:25c8:1946'))
    aaaab_response = dnsobj.qry(qtype='AAAA', resulttype='binary', timeout=1)
    self.assertTrue(aaaab_response.answers)
    self.assertEqual(len(aaaab_response.answers[0]['data']), 16)
    for b in aaaab_response.answers[0]['data']:
        assertIsByte(b)
    self.assertEqual(aaaab_response.answers[0]['data'], b'&\x06(\x00\x02 \x00\x01\x02H\x18\x93%\xc8\x19F')
    aaaai_response = dnsobj.qry(qtype='AAAA', resulttype='integer', timeout=1)
    self.assertTrue(aaaai_response.answers)
    self.assertEqual(aaaai_response.answers[0]['data'], 50542628918019813867414319910101719366)