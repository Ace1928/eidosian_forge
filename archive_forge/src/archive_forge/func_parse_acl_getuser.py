import datetime
from redis.utils import str_if_bytes
def parse_acl_getuser(response, **options):
    if response is None:
        return None
    if isinstance(response, list):
        data = pairs_to_dict(response, decode_keys=True)
    else:
        data = {str_if_bytes(key): value for key, value in response.items()}
    data['flags'] = list(map(str_if_bytes, data['flags']))
    data['passwords'] = list(map(str_if_bytes, data['passwords']))
    data['commands'] = str_if_bytes(data['commands'])
    if isinstance(data['keys'], str) or isinstance(data['keys'], bytes):
        data['keys'] = list(str_if_bytes(data['keys']).split(' '))
    if data['keys'] == ['']:
        data['keys'] = []
    if 'channels' in data:
        if isinstance(data['channels'], str) or isinstance(data['channels'], bytes):
            data['channels'] = list(str_if_bytes(data['channels']).split(' '))
        if data['channels'] == ['']:
            data['channels'] = []
    if 'selectors' in data:
        if data['selectors'] != [] and isinstance(data['selectors'][0], list):
            data['selectors'] = [list(map(str_if_bytes, selector)) for selector in data['selectors']]
        elif data['selectors'] != []:
            data['selectors'] = [{str_if_bytes(k): str_if_bytes(v) for k, v in selector.items()} for selector in data['selectors']]
    commands, categories = ([], [])
    for command in data['commands'].split(' '):
        categories.append(command) if '@' in command else commands.append(command)
    data['commands'] = commands
    data['categories'] = categories
    data['enabled'] = 'on' in data['flags']
    return data