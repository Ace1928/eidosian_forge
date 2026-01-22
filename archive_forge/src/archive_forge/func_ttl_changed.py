from __future__ import absolute_import, division, print_function
import traceback
from binascii import Error as binascii_error
from socket import error as socket_error
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def ttl_changed(self):
    query = dns.message.make_query(self.fqdn, self.module.params['type'])
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
    if lookup.rcode() != dns.rcode.NOERROR:
        self.module.fail_json(msg='Failed to lookup TTL of existing matching record.')
    current_ttl = lookup.answer[0].ttl if lookup.answer else lookup.authority[0].ttl
    return current_ttl != self.module.params['ttl']