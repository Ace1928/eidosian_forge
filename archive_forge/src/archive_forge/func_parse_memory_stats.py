import datetime
from redis.utils import str_if_bytes
def parse_memory_stats(response, **kwargs):
    """Parse the results of MEMORY STATS"""
    stats = pairs_to_dict(response, decode_keys=True, decode_string_values=True)
    for key, value in stats.items():
        if key.startswith('db.'):
            stats[key] = pairs_to_dict(value, decode_keys=True, decode_string_values=True)
    return stats