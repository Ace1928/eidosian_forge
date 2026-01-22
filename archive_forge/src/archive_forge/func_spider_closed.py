from email.utils import formatdate
from typing import Optional, Type, TypeVar
from twisted.internet import defer
from twisted.internet.error import (
from twisted.web.client import ResponseFailed
from scrapy import signals
from scrapy.crawler import Crawler
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http.request import Request
from scrapy.http.response import Response
from scrapy.settings import Settings
from scrapy.spiders import Spider
from scrapy.statscollectors import StatsCollector
from scrapy.utils.misc import load_object
def spider_closed(self, spider: Spider) -> None:
    self.storage.close_spider(spider)