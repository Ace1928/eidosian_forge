import logging
from collections import defaultdict
from scrapy import signals
from scrapy.exceptions import NotConfigured
def item_scraped_no_item(self, item, spider):
    self.items_in_period += 1