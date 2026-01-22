import datetime
from redis.utils import str_if_bytes
def parse_xpending(response, **options):
    if options.get('parse_detail', False):
        return parse_xpending_range(response)
    consumers = [{'name': n, 'pending': int(p)} for n, p in response[3] or []]
    return {'pending': response[0], 'min': response[1], 'max': response[2], 'consumers': consumers}