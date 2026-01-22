from twisted.web import http
from scrapy.exceptions import NotConfigured
from scrapy.utils.python import global_object_name, to_bytes
from scrapy.utils.request import request_httprepr
def get_header_size(headers):
    size = 0
    for key, value in headers.items():
        if isinstance(value, (list, tuple)):
            for v in value:
                size += len(b': ') + len(key) + len(v)
    return size + len(b'\r\n') * (len(headers.keys()) - 1)