from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible_collections.community.grafana.plugins.module_utils import base
from ansible.module_utils.six.moves.urllib.parse import quote
def is_user_update_required(target_user, email, name, login, is_admin):
    target_user_dict = dict(email=target_user.get('email'), name=target_user.get('name'), login=target_user.get('login'), is_admin=target_user.get('isGrafanaAdmin'))
    param_dict = dict(email=email, name=name, login=login, is_admin=is_admin)
    return target_user_dict != param_dict