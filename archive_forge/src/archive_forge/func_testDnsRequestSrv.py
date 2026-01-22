import DNS
import unittest
def testDnsRequestSrv(self):
    dnsobj = DNS.Request(qtype='srv')
    respdef = dnsobj.qry('_ldap._tcp.openldap.org', timeout=1)
    self.assertTrue(respdef.answers)
    data = respdef.answers[0]['data']
    self.assertEqual(len(data), 4)
    self.assertEqual(data[2], 389)
    self.assertTrue('openldap.org' in data[3])