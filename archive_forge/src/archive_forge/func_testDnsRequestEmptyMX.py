import DNS
import unittest
def testDnsRequestEmptyMX(self):
    dnsobj = DNS.DnsRequest('mail.kitterman.org')
    mx_empty_response = dnsobj.qry(qtype='MX', timeout=1)
    self.assertFalse(mx_empty_response.answers)