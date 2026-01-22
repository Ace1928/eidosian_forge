from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
def get_ssid(module, name, api_url, user, pwd):
    count = 0
    all_systems = 'storage-systems'
    systems_url = api_url + all_systems
    rc, data = request(systems_url, headers=HEADERS, url_username=user, url_password=pwd, validate_certs=module.validate_certs)
    for system in data:
        if system['name'] == name:
            count += 1
            if count > 1:
                module.fail_json(msg='You supplied a name for the Storage Array but more than 1 array was found with that name. ' + 'Use the id instead')
            else:
                ssid = system['id']
        else:
            continue
    if count == 0:
        module.fail_json(msg='No storage array with the name %s was found' % name)
    else:
        return ssid