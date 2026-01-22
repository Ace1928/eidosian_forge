import copy
import io
import json
import testtools
from urllib import parse
from glanceclient.v2 import schemas
def build_call_record(method, url, headers, data):
    """Key the request body be ordered if it's a dict type."""
    if isinstance(data, dict):
        data = sorted(data.items())
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return (method, url, headers or {}, data)
        data = [sorted(d.items()) for d in data]
    return (method, url, headers or {}, data)