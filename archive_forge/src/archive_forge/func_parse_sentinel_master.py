import datetime
from redis.utils import str_if_bytes
def parse_sentinel_master(response):
    return parse_sentinel_state(map(str_if_bytes, response))