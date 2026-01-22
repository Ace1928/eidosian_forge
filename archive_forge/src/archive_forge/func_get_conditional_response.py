import time
from collections import defaultdict
from hashlib import md5
from django.conf import settings
from django.core.cache import caches
from django.http import HttpResponse, HttpResponseNotModified
from django.utils.http import http_date, parse_etags, parse_http_date_safe, quote_etag
from django.utils.log import log_response
from django.utils.regex_helper import _lazy_re_compile
from django.utils.timezone import get_current_timezone_name
from django.utils.translation import get_language
def get_conditional_response(request, etag=None, last_modified=None, response=None):
    if response and (not 200 <= response.status_code < 300):
        return response
    if_match_etags = parse_etags(request.META.get('HTTP_IF_MATCH', ''))
    if_unmodified_since = request.META.get('HTTP_IF_UNMODIFIED_SINCE')
    if_unmodified_since = if_unmodified_since and parse_http_date_safe(if_unmodified_since)
    if_none_match_etags = parse_etags(request.META.get('HTTP_IF_NONE_MATCH', ''))
    if_modified_since = request.META.get('HTTP_IF_MODIFIED_SINCE')
    if_modified_since = if_modified_since and parse_http_date_safe(if_modified_since)
    if if_match_etags and (not _if_match_passes(etag, if_match_etags)):
        return _precondition_failed(request)
    if not if_match_etags and if_unmodified_since and (not _if_unmodified_since_passes(last_modified, if_unmodified_since)):
        return _precondition_failed(request)
    if if_none_match_etags and (not _if_none_match_passes(etag, if_none_match_etags)):
        if request.method in ('GET', 'HEAD'):
            return _not_modified(request, response)
        else:
            return _precondition_failed(request)
    if not if_none_match_etags and if_modified_since and (not _if_modified_since_passes(last_modified, if_modified_since)) and (request.method in ('GET', 'HEAD')):
        return _not_modified(request, response)
    return response