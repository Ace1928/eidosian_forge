import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
def check_for_exceptions(resp, body):
    if resp.status in (400, 422, 500):
        raise exceptions.from_response(resp, body)