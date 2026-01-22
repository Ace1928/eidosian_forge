import datetime
from redis.utils import str_if_bytes
def parse_sentinel_masters(response):
    result = {}
    for item in response:
        state = parse_sentinel_state(map(str_if_bytes, item))
        result[state['name']] = state
    return result