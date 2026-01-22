import copy
import urllib
from oslo_serialization import jsonutils
from keystoneauth1 import exceptions
from mistralclient import utils
def extract_json(response, response_key):
    if response_key is not None:
        return get_json(response)[response_key]
    else:
        return get_json(response)