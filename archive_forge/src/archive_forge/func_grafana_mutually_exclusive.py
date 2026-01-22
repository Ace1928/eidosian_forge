from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import url_argument_spec
def grafana_mutually_exclusive():
    return [['url_username', 'grafana_api_key']]