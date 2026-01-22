import datetime
from redis.utils import str_if_bytes
def string_keys_to_dict(key_string, callback):
    return dict.fromkeys(key_string.split(), callback)