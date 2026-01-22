import logging
from twisted.internet.defer import Deferred, maybeDeferred
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import Request
from scrapy.http.request import NO_CALLBACK
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import load_object
def process_request_2(self, rp, request, spider):
    if rp is None:
        return
    useragent = self._robotstxt_useragent
    if not useragent:
        useragent = request.headers.get(b'User-Agent', self._default_useragent)
    if not rp.allowed(request.url, useragent):
        logger.debug('Forbidden by robots.txt: %(request)s', {'request': request}, extra={'spider': spider})
        self.crawler.stats.inc_value('robotstxt/forbidden')
        raise IgnoreRequest('Forbidden by robots.txt')