from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def ospf_spec():
    return dict(area_cost=dict(type='int'), area_ctrl=dict(type='list', elements='str', choices=['redistribute', 'summary', 'suppress-fa', 'unspecified']), area_id=dict(type='str'), area_type=dict(type='str', choices=['nssa', 'regular', 'stub']), description=dict(type='str', aliases=['descr']), multipod_internal=dict(type='str', choices=['no', 'yes']), name_alias=dict(type='str'))