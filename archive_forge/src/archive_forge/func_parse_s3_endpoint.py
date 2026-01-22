import string
from urllib.parse import urlparse
from ansible.module_utils.basic import to_text
def parse_s3_endpoint(options):
    endpoint_url = options.get('endpoint_url')
    if options.get('ceph'):
        return (False, parse_ceph_endpoint(endpoint_url))
    if is_fakes3(endpoint_url):
        return (False, parse_fakes3_endpoint(endpoint_url))
    return (True, {'endpoint': endpoint_url})