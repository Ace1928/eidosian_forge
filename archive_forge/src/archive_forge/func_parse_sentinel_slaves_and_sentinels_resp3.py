import datetime
from redis.utils import str_if_bytes
def parse_sentinel_slaves_and_sentinels_resp3(response):
    return [parse_sentinel_state_resp3(item) for item in response]