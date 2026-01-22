from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
def get_all_services(module, auth):
    try:
        response = open_url(auth.url + '/service', method='GET', force_basic_auth=True, url_username=auth.user, url_password=auth.password)
    except Exception as e:
        module.fail_json(msg=str(e))
    return module.from_json(response.read())