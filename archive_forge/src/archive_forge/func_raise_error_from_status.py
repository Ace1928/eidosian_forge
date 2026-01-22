import httplib2
import logging
import os
import sys
import time
from troveclient.compat import auth
from troveclient.compat import exceptions
def raise_error_from_status(self, resp, body):
    if resp.status in expected_errors:
        raise exceptions.from_response(resp, body)