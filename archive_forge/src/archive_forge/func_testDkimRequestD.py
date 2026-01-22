import DNS
import unittest
def testDkimRequestD(self):
    q = '20161025._domainkey.google.com'
    dnsob = DNS.Request(q, qtype='txt')
    resp = dnsob.req(timeout=1)
    self.assertTrue(resp.answers)
    data = resp.answers[0]['data']
    self.assertFalse(isinstance(data[0], str))
    self.assertTrue(data[0].startswith(b'k=rsa'))