import datetime
from redis.utils import str_if_bytes
def timestamp_to_datetime(response):
    """Converts a unix timestamp to a Python datetime object"""
    if not response:
        return None
    try:
        response = int(response)
    except ValueError:
        return None
    return datetime.datetime.fromtimestamp(response)