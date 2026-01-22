from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.six import integer_types, string_types
from ansible.module_utils.six.moves import configparser
from ansible.module_utils.urls import fetch_url
def exo_dns_required_together():
    return [['api_key', 'api_secret']]