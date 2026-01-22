from django.utils.cache import cc_delim_re, get_conditional_response, set_response_etag
from django.utils.deprecation import MiddlewareMixin
from django.utils.http import parse_http_date_safe
def needs_etag(self, response):
    """Return True if an ETag header should be added to response."""
    cache_control_headers = cc_delim_re.split(response.get('Cache-Control', ''))
    return all((header.lower() != 'no-store' for header in cache_control_headers))