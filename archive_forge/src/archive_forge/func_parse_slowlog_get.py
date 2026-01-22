import datetime
from redis.utils import str_if_bytes
def parse_slowlog_get(response, **options):
    space = ' ' if options.get('decode_responses', False) else b' '

    def parse_item(item):
        result = {'id': item[0], 'start_time': int(item[1]), 'duration': int(item[2])}
        if isinstance(item[3], list):
            result['command'] = space.join(item[3])
            result['client_address'] = item[4]
            result['client_name'] = item[5]
        else:
            result['complexity'] = item[3]
            result['command'] = space.join(item[4])
            result['client_address'] = item[5]
            result['client_name'] = item[6]
        return result
    return [parse_item(item) for item in response]