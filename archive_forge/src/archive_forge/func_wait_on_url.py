from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.urls import Request, SSLValidationError, ConnectionError
from ansible.module_utils.parsing.convert_bool import boolean as strtobool
from ansible.module_utils.six import PY2
from ansible.module_utils.six import raise_from, string_types
from ansible.module_utils.six.moves import StringIO
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.http_cookiejar import CookieJar
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode, quote
from ansible.module_utils.six.moves.configparser import ConfigParser, NoOptionError
from socket import getaddrinfo, IPPROTO_TCP
import time
import re
from json import loads, dumps
from os.path import isfile, expanduser, split, join, exists, isdir
from os import access, R_OK, getcwd, environ
def wait_on_url(self, url, object_name, object_type, timeout=30, interval=2):
    start = time.time()
    result = self.get_endpoint(url)
    while not result['json']['finished']:
        if timeout and timeout < time.time() - start:
            if object_type == 'legacy_job_wait':
                self.json_output['msg'] = 'Monitoring of Job - {0} aborted due to timeout'.format(object_name)
            else:
                self.json_output['msg'] = 'Monitoring of {0} - {1} aborted due to timeout'.format(object_type, object_name)
            self.wait_output(result)
            self.fail_json(**self.json_output)
        time.sleep(interval)
        result = self.get_endpoint(url)
        self.json_output['status'] = result['json']['status']
    if result['json']['failed']:
        if object_type == 'legacy_job_wait':
            self.json_output['msg'] = 'Job with id {0} failed'.format(object_name)
        else:
            self.json_output['msg'] = 'The {0} - {1}, failed'.format(object_type, object_name)
            self.json_output['job_data'] = result['json']
        self.wait_output(result)
        self.fail_json(**self.json_output)
    self.wait_output(result)
    return result