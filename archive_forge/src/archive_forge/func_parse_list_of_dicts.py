import datetime
from redis.utils import str_if_bytes
def parse_list_of_dicts(response):
    return list(map(pairs_to_dict_with_str_keys, response))