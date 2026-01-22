import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
def limit_url(url, limit=None, marker=None):
    if not limit and (not marker):
        return url
    query = []
    if marker:
        query.append('marker=%s' % marker)
    if limit:
        query.append('limit=%s' % limit)
    query = '?' + '&'.join(query)
    return url + query