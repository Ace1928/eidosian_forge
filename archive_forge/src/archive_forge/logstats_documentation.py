import logging
from twisted.internet import task
from scrapy import signals
from scrapy.exceptions import NotConfigured
Log basic scraping stats periodically