import os
import signal
from itemadapter import is_item
from twisted.internet import defer, threads
from twisted.python import threadable
from w3lib.url import any_to_uri
from scrapy.crawler import Crawler
from scrapy.exceptions import IgnoreRequest
from scrapy.http import Request, Response
from scrapy.settings import Settings
from scrapy.spiders import Spider
from scrapy.utils.conf import get_config
from scrapy.utils.console import DEFAULT_PYTHON_SHELLS, start_python_console
from scrapy.utils.datatypes import SequenceExclude
from scrapy.utils.misc import load_object
from scrapy.utils.reactor import is_asyncio_reactor_installed, set_asyncio_event_loop
from scrapy.utils.response import open_in_browser
def populate_vars(self, response=None, request=None, spider=None):
    import scrapy
    self.vars['scrapy'] = scrapy
    self.vars['crawler'] = self.crawler
    self.vars['item'] = self.item_class()
    self.vars['settings'] = self.crawler.settings
    self.vars['spider'] = spider
    self.vars['request'] = request
    self.vars['response'] = response
    if self.inthread:
        self.vars['fetch'] = self.fetch
    self.vars['view'] = open_in_browser
    self.vars['shelp'] = self.print_help
    self.update_vars(self.vars)
    if not self.code:
        self.vars['banner'] = self.get_help()