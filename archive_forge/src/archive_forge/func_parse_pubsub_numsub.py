import datetime
from redis.utils import str_if_bytes
def parse_pubsub_numsub(response, **options):
    return list(zip(response[0::2], response[1::2]))