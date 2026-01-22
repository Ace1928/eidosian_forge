import datetime
from redis.utils import str_if_bytes
def parse_acl_log(response, **options):
    if response is None:
        return None
    if isinstance(response, list):
        data = []
        for log in response:
            log_data = pairs_to_dict(log, True, True)
            client_info = log_data.get('client-info', '')
            log_data['client-info'] = parse_client_info(client_info)
            log_data['age-seconds'] = float(log_data['age-seconds'])
            data.append(log_data)
    else:
        data = bool_ok(response)
    return data