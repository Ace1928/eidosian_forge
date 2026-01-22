from datetime import datetime, timezone
from scrapy import signals
def item_dropped(self, item, spider, exception):
    reason = exception.__class__.__name__
    self.stats.inc_value('item_dropped_count', spider=spider)
    self.stats.inc_value(f'item_dropped_reasons_count/{reason}', spider=spider)