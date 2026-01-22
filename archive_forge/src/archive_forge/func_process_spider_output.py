import logging
from scrapy.http import Request
def process_spider_output(self, response, result, spider):
    self._init_depth(response, spider)
    return (r for r in result or () if self._filter(r, response, spider))