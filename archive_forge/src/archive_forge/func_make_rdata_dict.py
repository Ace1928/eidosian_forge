from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.utils.display import Display
import socket
def make_rdata_dict(rdata):
    """ While the 'dig' lookup plugin supports anything which dnspython supports
        out of the box, the following supported_types list describes which
        DNS query types we can convert to a dict.

        Note: adding support for RRSIG is hard work. :)
    """
    supported_types = {A: ['address'], AAAA: ['address'], CAA: ['flags', 'tag', 'value'], CNAME: ['target'], DNAME: ['target'], DNSKEY: ['flags', 'algorithm', 'protocol', 'key'], DS: ['algorithm', 'digest_type', 'key_tag', 'digest'], HINFO: ['cpu', 'os'], LOC: ['latitude', 'longitude', 'altitude', 'size', 'horizontal_precision', 'vertical_precision'], MX: ['preference', 'exchange'], NAPTR: ['order', 'preference', 'flags', 'service', 'regexp', 'replacement'], NS: ['target'], NSEC3PARAM: ['algorithm', 'flags', 'iterations', 'salt'], PTR: ['target'], RP: ['mbox', 'txt'], SOA: ['mname', 'rname', 'serial', 'refresh', 'retry', 'expire', 'minimum'], SPF: ['strings'], SRV: ['priority', 'weight', 'port', 'target'], SSHFP: ['algorithm', 'fp_type', 'fingerprint'], TLSA: ['usage', 'selector', 'mtype', 'cert'], TXT: ['strings']}
    rd = {}
    if rdata.rdtype in supported_types:
        fields = supported_types[rdata.rdtype]
        for f in fields:
            val = rdata.__getattribute__(f)
            if isinstance(val, dns.name.Name):
                val = dns.name.Name.to_text(val)
            if rdata.rdtype == DS and f == 'digest':
                val = dns.rdata._hexify(rdata.digest).replace(' ', '')
            if rdata.rdtype == DNSKEY and f == 'algorithm':
                val = int(val)
            if rdata.rdtype == DNSKEY and f == 'key':
                val = dns.rdata._base64ify(rdata.key).replace(' ', '')
            if rdata.rdtype == NSEC3PARAM and f == 'salt':
                val = dns.rdata._hexify(rdata.salt).replace(' ', '')
            if rdata.rdtype == SSHFP and f == 'fingerprint':
                val = dns.rdata._hexify(rdata.fingerprint).replace(' ', '')
            if rdata.rdtype == TLSA and f == 'cert':
                val = dns.rdata._hexify(rdata.cert).replace(' ', '')
            rd[f] = val
    return rd