import io
import logging
import urllib.parse
from smart_open import utils, constants
import http.client as httplib
def convert_to_http_uri(parsed_uri):
    return _convert_to_http_uri(parsed_uri.uri)