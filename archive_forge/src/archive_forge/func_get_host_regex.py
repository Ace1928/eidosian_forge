import logging
import re
import warnings
from scrapy import signals
from scrapy.http import Request
from scrapy.utils.httpobj import urlparse_cached
def get_host_regex(self, spider):
    """Override this method to implement a different offsite policy"""
    allowed_domains = getattr(spider, 'allowed_domains', None)
    if not allowed_domains:
        return re.compile('')
    url_pattern = re.compile('^https?://.*$')
    port_pattern = re.compile(':\\d+$')
    domains = []
    for domain in allowed_domains:
        if domain is None:
            continue
        if url_pattern.match(domain):
            message = f'allowed_domains accepts only domains, not URLs. Ignoring URL entry {domain} in allowed_domains.'
            warnings.warn(message, URLWarning)
        elif port_pattern.search(domain):
            message = f'allowed_domains accepts only domains without ports. Ignoring entry {domain} in allowed_domains.'
            warnings.warn(message, PortWarning)
        else:
            domains.append(re.escape(domain))
    regex = f'^(.*\\.)?({'|'.join(domains)})$'
    return re.compile(regex)