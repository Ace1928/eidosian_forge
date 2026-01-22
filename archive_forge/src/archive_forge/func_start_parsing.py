import inspect
import json
import logging
from typing import Dict
from itemadapter import ItemAdapter, is_item
from twisted.internet.defer import maybeDeferred
from w3lib.url import is_url
from scrapy.commands import BaseRunSpiderCommand
from scrapy.exceptions import UsageError
from scrapy.http import Request
from scrapy.utils import display
from scrapy.utils.asyncgen import collect_asyncgen
from scrapy.utils.defer import aiter_errback, deferred_from_coro
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import arg_to_iter
from scrapy.utils.spider import spidercls_for_request
def start_parsing(self, url, opts):
    self.crawler_process.crawl(self.spidercls, **opts.spargs)
    self.pcrawler = list(self.crawler_process.crawlers)[0]
    self.crawler_process.start()
    if not self.first_response:
        logger.error('No response downloaded for: %(url)s', {'url': url})