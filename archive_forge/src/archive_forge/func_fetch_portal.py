from __future__ import absolute_import, division, print_function
import json
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text, to_native
def fetch_portal(module, path, method, data=None, accept_header='application/json'):
    if data is None:
        data = {}
    url = module.params['portal'] + path
    headers = {'accept': accept_header, 'content-type': 'application/json'}
    fetch_kwargs = {'timeout': 30}
    if os.path.exists(REDHAT_UEP):
        fetch_kwargs['ca_path'] = REDHAT_UEP
    try:
        resp, info = fetch_url(module, url, json.dumps(data), headers, method, **fetch_kwargs)
    except TypeError:
        if module.params['validate_certs']:
            module.warn('Your Ansible version does not support providing custom CA certificates for HTTP requests. Talking to the Red Hat portal might fail without validate_certs=False. Please update.')
        del fetch_kwargs['ca_path']
        resp, info = fetch_url(module, url, json.dumps(data), headers, method, **fetch_kwargs)
    if resp is None or info['status'] >= 400:
        try:
            error = json.loads(info['body'])['displayMessage']
        except Exception:
            error = info['msg']
        module.fail_json(msg='%s to %s failed, got %s' % (method, url, error))
    return (resp, info)