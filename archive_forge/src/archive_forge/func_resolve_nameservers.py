from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
def resolve_nameservers(self, target, resolve_addresses=False):
    nameservers = self._lookup_ns(dns.name.from_unicode(to_text(target)))
    if resolve_addresses:
        nameserver_ips = set()
        for nameserver in nameservers or []:
            nameserver_ips.update(self._lookup_address(nameserver))
        nameservers = list(nameserver_ips)
    return sorted(nameservers or [])