import string
from urllib.parse import urlparse
from ansible.module_utils.basic import to_text
def is_fakes3(url):
    """Return True if endpoint_url has scheme fakes3://"""
    result = False
    if url is not None:
        result = urlparse(url).scheme in ('fakes3', 'fakes3s')
    return result