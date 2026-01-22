from __future__ import absolute_import, division, print_function
import random
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import fetch_url
def vultr_argument_spec():
    return dict(api_endpoint=dict(type='str', fallback=(env_fallback, ['VULTR_API_ENDPOINT']), default='https://api.vultr.com/v2'), api_key=dict(type='str', fallback=(env_fallback, ['VULTR_API_KEY']), no_log=True, required=True), api_timeout=dict(type='int', fallback=(env_fallback, ['VULTR_API_TIMEOUT']), default=180), api_retries=dict(type='int', fallback=(env_fallback, ['VULTR_API_RETRIES']), default=5), api_retry_max_delay=dict(type='int', fallback=(env_fallback, ['VULTR_API_RETRY_MAX_DELAY']), default=12), validate_certs=dict(type='bool', default=True))