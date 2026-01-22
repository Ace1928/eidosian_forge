import datetime
from redis.utils import str_if_bytes
def float_or_none(response):
    if response is None:
        return None
    return float(response)