from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def get_api_config(self):
    api_config = {'endpoint': self.module.params.get('api_url'), 'key': self.module.params.get('api_key'), 'secret': self.module.params.get('api_secret'), 'timeout': self.module.params.get('api_timeout'), 'method': self.module.params.get('api_http_method'), 'verify': self.module.params.get('api_verify_ssl_cert')}
    self.result.update({'api_url': api_config['endpoint'], 'api_key': api_config['key'], 'api_timeout': int(api_config['timeout']), 'api_http_method': api_config['method'], 'api_verify_ssl_cert': api_config['verify']})
    return api_config