import httplib2
import logging
import os
import sys
import time
from troveclient.compat import auth
from troveclient.compat import exceptions
def morph_response_body(self, raw_body):
    try:
        return json.loads(raw_body.decode())
    except ValueError:
        raise exceptions.ResponseFormatError()