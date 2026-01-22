from __future__ import absolute_import, division, print_function
import traceback
from binascii import Error as binascii_error
from socket import error as socket_error
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def lookup_zone(self):
    name = dns.name.from_text(self.module.params['record'])
    while True:
        query = dns.message.make_query(name, dns.rdatatype.SOA)
        if self.keyring:
            query.use_tsig(keyring=self.keyring, algorithm=self.algorithm)
        try:
            if self.module.params['protocol'] == 'tcp':
                lookup = dns.query.tcp(query, self.module.params['server'], timeout=10, port=self.module.params['port'])
            else:
                lookup = dns.query.udp(query, self.module.params['server'], timeout=10, port=self.module.params['port'])
        except (dns.tsig.PeerBadKey, dns.tsig.PeerBadSignature) as e:
            self.module.fail_json(msg='TSIG update error (%s): %s' % (e.__class__.__name__, to_native(e)))
        except (socket_error, dns.exception.Timeout) as e:
            self.module.fail_json(msg='DNS server error: (%s): %s' % (e.__class__.__name__, to_native(e)))
        if lookup.rcode() in [dns.rcode.SERVFAIL, dns.rcode.REFUSED]:
            self.module.fail_json(msg="Zone lookup failure: '%s' will not respond to queries regarding '%s'." % (self.module.params['server'], self.module.params['record']))
        for rr in lookup.answer:
            if rr.rdtype == dns.rdatatype.SOA and rr.name == name:
                return rr.name.to_text()
        for rr in lookup.authority:
            if rr.rdtype == dns.rdatatype.SOA and name.fullcompare(rr.name)[0] == dns.name.NAMERELN_SUBDOMAIN:
                return rr.name.to_text()
        try:
            name = name.parent()
        except dns.name.NoParent:
            self.module.fail_json(msg="Zone lookup of '%s' failed for unknown reason." % self.module.params['record'])