import logging
from datetime import datetime, timezone
from twisted.internet import task
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.utils.serialize import ScrapyJSONEncoder
def log_crawler_stats(self):
    stats = {k: v for k, v in self.stats._stats.items() if self.param_allowed(k, self.ext_stats_include, self.ext_stats_exclude)}
    return {'stats': stats}