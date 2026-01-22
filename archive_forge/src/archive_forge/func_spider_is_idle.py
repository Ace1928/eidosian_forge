import logging
from time import time
from typing import (
from twisted.internet.defer import Deferred, inlineCallbacks, succeed
from twisted.internet.task import LoopingCall
from twisted.python.failure import Failure
from scrapy import signals
from scrapy.core.downloader import Downloader
from scrapy.core.scraper import Scraper
from scrapy.exceptions import CloseSpider, DontCloseSpider
from scrapy.http import Request, Response
from scrapy.logformatter import LogFormatter
from scrapy.settings import BaseSettings, Settings
from scrapy.signalmanager import SignalManager
from scrapy.spiders import Spider
from scrapy.utils.log import failure_to_exc_info, logformatter_adapter
from scrapy.utils.misc import create_instance, load_object
from scrapy.utils.reactor import CallLaterOnce
def spider_is_idle(self) -> bool:
    if self.slot is None:
        raise RuntimeError('Engine slot not assigned')
    if not self.scraper.slot.is_idle():
        return False
    if self.downloader.active:
        return False
    if self.slot.start_requests is not None:
        return False
    if self.slot.scheduler.has_pending_requests():
        return False
    return True