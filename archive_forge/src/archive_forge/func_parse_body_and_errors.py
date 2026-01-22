import base64
from libcloud.utils.py3 import b
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError
def parse_body_and_errors(self):
    data = []
    errors = []
    js = super().parse_body()
    if 'items' in js:
        data.append(js['items'])
    if 'name' in js:
        data.append(js)
    if 'deleted' in js:
        data.append(js['deleted'])
    if 'error_class' in js:
        errors.append(js)
    return (data, errors)