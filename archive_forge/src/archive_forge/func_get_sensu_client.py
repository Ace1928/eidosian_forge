from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import env_fallback
from . import client
def get_sensu_client(auth):
    return client.Client(auth['url'], auth['user'], auth['password'], auth['api_key'], auth['verify'], auth['ca_path'])