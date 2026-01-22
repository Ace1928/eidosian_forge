from __future__ import (absolute_import, division, print_function)
import logging
import time
from ansible.module_utils.basic import missing_required_lib
def cloudmanager_host_argument_spec():
    return dict(refresh_token=dict(required=False, type='str', no_log=True), sa_client_id=dict(required=False, type='str', no_log=True), sa_secret_key=dict(required=False, type='str', no_log=True), environment=dict(required=False, type='str', choices=['prod', 'stage'], default='prod'), feature_flags=dict(required=False, type='dict'))