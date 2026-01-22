import string
from urllib.parse import urlparse
from ansible.module_utils.basic import to_text
def s3_extra_params(options, sigv4=False):
    aws, extra_params = parse_s3_endpoint(options)
    endpoint = extra_params['endpoint']
    if not aws:
        return extra_params
    dualstack = options.get('dualstack')
    if not dualstack and (not sigv4):
        return extra_params
    config = {}
    if dualstack:
        config['use_dualstack_endpoint'] = True
    if sigv4:
        config['signature_version'] = 's3v4'
    extra_params['config'] = config
    return extra_params