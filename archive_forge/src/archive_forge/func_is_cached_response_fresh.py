import gzip
import logging
import pickle
from email.utils import mktime_tz, parsedate_tz
from importlib import import_module
from pathlib import Path
from time import time
from weakref import WeakKeyDictionary
from w3lib.http import headers_dict_to_raw, headers_raw_to_dict
from scrapy.http import Headers, Response
from scrapy.http.request import Request
from scrapy.responsetypes import responsetypes
from scrapy.spiders import Spider
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.project import data_path
from scrapy.utils.python import to_bytes, to_unicode
def is_cached_response_fresh(self, cachedresponse, request):
    cc = self._parse_cachecontrol(cachedresponse)
    ccreq = self._parse_cachecontrol(request)
    if b'no-cache' in cc or b'no-cache' in ccreq:
        return False
    now = time()
    freshnesslifetime = self._compute_freshness_lifetime(cachedresponse, request, now)
    currentage = self._compute_current_age(cachedresponse, request, now)
    reqmaxage = self._get_max_age(ccreq)
    if reqmaxage is not None:
        freshnesslifetime = min(freshnesslifetime, reqmaxage)
    if currentage < freshnesslifetime:
        return True
    if b'max-stale' in ccreq and b'must-revalidate' not in cc:
        staleage = ccreq[b'max-stale']
        if staleage is None:
            return True
        try:
            if currentage < freshnesslifetime + max(0, int(staleage)):
                return True
        except ValueError:
            pass
    self._set_conditional_validators(request, cachedresponse)
    return False