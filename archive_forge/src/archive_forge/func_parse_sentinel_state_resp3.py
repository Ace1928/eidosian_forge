import datetime
from redis.utils import str_if_bytes
def parse_sentinel_state_resp3(response):
    result = {}
    for key in response:
        try:
            value = SENTINEL_STATE_TYPES[key](str_if_bytes(response[key]))
            result[str_if_bytes(key)] = value
        except Exception:
            result[str_if_bytes(key)] = response[str_if_bytes(key)]
    flags = set(result['flags'].split(','))
    result['flags'] = flags
    return result