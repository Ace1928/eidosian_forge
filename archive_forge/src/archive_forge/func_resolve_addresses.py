from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
def resolve_addresses(self, target, **kwargs):
    dnsname = dns.name.from_unicode(to_text(target))
    resolver = self.default_resolver
    result = []
    try:
        for data in self._resolve(resolver, dnsname, handle_response_errors=True, rdtype=dns.rdatatype.A, **kwargs):
            result.append(str(data))
    except dns.resolver.NoAnswer:
        pass
    try:
        for data in self._resolve(resolver, dnsname, handle_response_errors=True, rdtype=dns.rdatatype.AAAA, **kwargs):
            result.append(str(data))
    except dns.resolver.NoAnswer:
        pass
    return result