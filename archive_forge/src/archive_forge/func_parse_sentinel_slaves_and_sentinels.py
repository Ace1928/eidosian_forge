import datetime
from redis.utils import str_if_bytes
def parse_sentinel_slaves_and_sentinels(response):
    return [parse_sentinel_state(map(str_if_bytes, item)) for item in response]