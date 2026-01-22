from boto.compat import http_client
from tests.compat import mock, unittest
def overwrite_header(arg, default=None):
    header_dict = dict(header)
    if arg in header_dict:
        return header_dict[arg]
    else:
        return default