from __future__ import annotations
import logging
from collections import deque
from typing import (
from itemadapter import is_item
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure
from scrapy import Spider, signals
from scrapy.core.spidermw import SpiderMiddlewareManager
from scrapy.exceptions import CloseSpider, DropItem, IgnoreRequest
from scrapy.http import Request, Response
from scrapy.logformatter import LogFormatter
from scrapy.pipelines import ItemPipelineManager
from scrapy.signalmanager import SignalManager
from scrapy.utils.defer import (
from scrapy.utils.log import failure_to_exc_info, logformatter_adapter
from scrapy.utils.misc import load_object, warn_on_generator_with_return_value
from scrapy.utils.spider import iterate_spider_output
def handle_spider_error(self, _failure: Failure, request: Request, response: Union[Response, Failure], spider: Spider) -> None:
    exc = _failure.value
    if isinstance(exc, CloseSpider):
        assert self.crawler.engine is not None
        self.crawler.engine.close_spider(spider, exc.reason or 'cancelled')
        return
    logkws = self.logformatter.spider_error(_failure, request, response, spider)
    logger.log(*logformatter_adapter(logkws), exc_info=failure_to_exc_info(_failure), extra={'spider': spider})
    self.signals.send_catch_log(signal=signals.spider_error, failure=_failure, response=response, spider=spider)
    assert self.crawler.stats
    self.crawler.stats.inc_value(f'spider_exceptions/{_failure.value.__class__.__name__}', spider=spider)