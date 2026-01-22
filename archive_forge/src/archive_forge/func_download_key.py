from __future__ import absolute_import, division, print_function
import os
from traceback import format_exc
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.urls import fetch_url
def download_key(module, url):
    try:
        rsp, info = fetch_url(module, url, use_proxy=True)
        if info['status'] != 200:
            module.fail_json(msg='Failed to download key at %s: %s' % (url, info['msg']))
        return rsp.read()
    except Exception:
        module.fail_json(msg='error getting key id from url: %s' % url, traceback=format_exc())