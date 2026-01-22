from scrapy import signals
def spider_opened(self, spider):
    self.user_agent = getattr(spider, 'user_agent', self.user_agent)