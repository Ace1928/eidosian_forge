import DNS
import unittest
def testNSD(self):
    """Lookup NS record from SOA"""
    dnsob = DNS.DnsRequest('kitterman.com')
    resp = dnsob.req(qtype='SOA', timeout=1)
    self.assertTrue(resp.answers)
    primary = resp.answers[0]['data'][0]
    self.assertEqual(primary, 'ns1.pairnic.com')
    resp = dnsob.req(qtype='NS', server=primary, aa=1, timeout=1)
    nslist = [x['data'].lower() for x in resp.answers]
    nslist.sort()
    self.assertEqual(nslist, ['ns1.pairnic.com', 'ns2.pairnic.com'])