import DNS
import unittest
def testDnsRequestEmptyMXD(self):
    dnsob = DNS.DnsRequest('mail.kitterman.org')
    mx_empty_response = dnsob.req(qtype='MX', timeout=1)
    self.assertFalse(mx_empty_response.answers)