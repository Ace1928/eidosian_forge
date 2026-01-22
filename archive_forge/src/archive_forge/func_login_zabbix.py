from __future__ import print_function
import os
import sys
import argparse
import json
import atexit
from ansible.module_utils.six.moves import configparser
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import Request
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def login_zabbix(self):
    auth_token = self.auth_token
    if auth_token:
        self.auth = auth_token
        return
    atexit.register(self.logout_zabbix)
    login_user = self.zabbix_username
    login_password = self.zabbix_password
    response = self.api_request('user.login', {'username': login_user, 'password': login_password})
    res = json.load(response)
    self.auth = res['result']