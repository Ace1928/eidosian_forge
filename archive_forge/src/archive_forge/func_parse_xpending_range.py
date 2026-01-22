import datetime
from redis.utils import str_if_bytes
def parse_xpending_range(response):
    k = ('message_id', 'consumer', 'time_since_delivered', 'times_delivered')
    return [dict(zip(k, r)) for r in response]