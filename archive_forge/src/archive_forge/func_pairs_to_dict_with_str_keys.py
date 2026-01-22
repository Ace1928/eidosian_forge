import datetime
from redis.utils import str_if_bytes
def pairs_to_dict_with_str_keys(response):
    return pairs_to_dict(response, decode_keys=True)