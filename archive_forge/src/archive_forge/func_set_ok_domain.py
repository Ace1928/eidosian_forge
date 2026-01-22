import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def set_ok_domain(self, cookie, request):
    if self.is_blocked(cookie.domain):
        _debug('   domain %s is in user block-list', cookie.domain)
        return False
    if self.is_not_allowed(cookie.domain):
        _debug('   domain %s is not in user allow-list', cookie.domain)
        return False
    if cookie.domain_specified:
        req_host, erhn = eff_request_host(request)
        domain = cookie.domain
        if self.strict_domain and domain.count('.') >= 2:
            i = domain.rfind('.')
            j = domain.rfind('.', 0, i)
            if j == 0:
                tld = domain[i + 1:]
                sld = domain[j + 1:i]
                if sld.lower() in ('co', 'ac', 'com', 'edu', 'org', 'net', 'gov', 'mil', 'int', 'aero', 'biz', 'cat', 'coop', 'info', 'jobs', 'mobi', 'museum', 'name', 'pro', 'travel', 'eu') and len(tld) == 2:
                    _debug('   country-code second level domain %s', domain)
                    return False
        if domain.startswith('.'):
            undotted_domain = domain[1:]
        else:
            undotted_domain = domain
        embedded_dots = undotted_domain.find('.') >= 0
        if not embedded_dots and (not erhn.endswith('.local')):
            _debug('   non-local domain %s contains no embedded dot', domain)
            return False
        if cookie.version == 0:
            if not (erhn.endswith(domain) or erhn.endswith(f'{undotted_domain}.local')) and (not erhn.startswith('.') and (not ('.' + erhn).endswith(domain))):
                _debug('   effective request-host %s (even with added initial dot) does not end with %s', erhn, domain)
                return False
        if cookie.version > 0 or self.strict_ns_domain & self.DomainRFC2965Match:
            if not domain_match(erhn, domain):
                _debug('   effective request-host %s does not domain-match %s', erhn, domain)
                return False
        if cookie.version > 0 or self.strict_ns_domain & self.DomainStrictNoDots:
            host_prefix = req_host[:-len(domain)]
            if host_prefix.find('.') >= 0 and (not IPV4_RE.search(req_host)):
                _debug('   host prefix %s for domain %s contains a dot', host_prefix, domain)
                return False
    return True