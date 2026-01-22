from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import url_argument_spec
def grafana_argument_spec():
    argument_spec = url_argument_spec()
    del argument_spec['force']
    del argument_spec['force_basic_auth']
    del argument_spec['http_agent']
    if 'use_gssapi' in argument_spec:
        del argument_spec['use_gssapi']
    argument_spec.update(state=dict(choices=['present', 'absent'], default='present'), url=dict(aliases=['grafana_url'], type='str', required=True), grafana_api_key=dict(type='str', no_log=True), url_username=dict(aliases=['grafana_user'], default='admin'), url_password=dict(aliases=['grafana_password'], default='admin', no_log=True))
    return argument_spec