from __future__ import (absolute_import, division, print_function)
from base64 import b64encode
from email.utils import formatdate
import re
import json
import hashlib
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.basic import env_fallback
def get_moid_by_name(self, resource_path, target_name):
    """
        Retrieve an Intersight object moid by name

        :param resource_path: intersight resource path e.g. '/ntp/Policies'
        :param target_name: intersight object name
        :return: json http response object
        """
    query_params = {'$filter': "Name eq '{0}'".format(target_name)}
    options = {'http_method': 'GET', 'resource_path': resource_path, 'query_params': query_params}
    get_moid = self.intersight_call(**options)
    if get_moid.json()['Results'] is not None:
        located_moid = get_moid.json()['Results'][0]['Moid']
    else:
        raise KeyError('Intersight object with name "{0}" not found!'.format(target_name))
    return located_moid