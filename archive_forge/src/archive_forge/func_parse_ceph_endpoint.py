import string
from urllib.parse import urlparse
from ansible.module_utils.basic import to_text
def parse_ceph_endpoint(url):
    ceph = urlparse(url)
    use_ssl = bool(ceph.scheme == 'https')
    return {'endpoint': url, 'use_ssl': use_ssl}