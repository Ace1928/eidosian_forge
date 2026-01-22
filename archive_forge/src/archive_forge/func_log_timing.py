import logging
from datetime import datetime, timezone
from twisted.internet import task
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.utils.serialize import ScrapyJSONEncoder
def log_timing(self):
    now = datetime.now(tz=timezone.utc)
    time = {'log_interval': self.interval, 'start_time': self.stats._stats['start_time'], 'utcnow': now, 'log_interval_real': (now - self.time_prev).total_seconds(), 'elapsed': (now - self.stats._stats['start_time']).total_seconds()}
    self.time_prev = now
    return {'time': time}