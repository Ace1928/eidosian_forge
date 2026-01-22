from __future__ import absolute_import, division, print_function
import json
import random
import mimetypes
from pprint import pformat
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils._text import to_native
def is_web_services_version_met(self, version):
    """Determines whether a particular web services version has been satisfied."""
    split_version = version.split('.')
    if len(split_version) != 4 or not split_version[0].isdigit() or (not split_version[1].isdigit()) or (not split_version[3].isdigit()):
        self.module.fail_json(msg='Version is not a valid Web Services version. Version [%s].' % version)
    url_parts = urlparse(self.url)
    if not url_parts.scheme or not url_parts.netloc:
        self.module.fail_json(msg='Failed to provide valid API URL. Example: https://192.168.1.100:8443/devmgr/v2. URL [%s].' % self.url)
    if url_parts.scheme not in ['http', 'https']:
        self.module.fail_json(msg='Protocol must be http or https. URL [%s].' % self.url)
    self.url = '%s://%s/' % (url_parts.scheme, url_parts.netloc)
    about_url = self.url + self.DEFAULT_REST_API_ABOUT_PATH
    rc, data = request(about_url, timeout=self.DEFAULT_TIMEOUT, headers=self.DEFAULT_HEADERS, ignore_errors=True, **self.creds)
    if rc != 200:
        self.module.warn('Failed to retrieve web services about information! Retrying with secure ports. Array Id [%s].' % self.ssid)
        self.url = 'https://%s:8443/' % url_parts.netloc.split(':')[0]
        about_url = self.url + self.DEFAULT_REST_API_ABOUT_PATH
        try:
            rc, data = request(about_url, timeout=self.DEFAULT_TIMEOUT, headers=self.DEFAULT_HEADERS, **self.creds)
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve the webservices about information! Array Id [%s]. Error [%s].' % (self.ssid, to_native(error)))
    if len(data['version'].split('.')) == 4:
        major, minor, other, revision = data['version'].split('.')
        minimum_major, minimum_minor, other, minimum_revision = split_version
        if not (major > minimum_major or (major == minimum_major and minor > minimum_minor) or (major == minimum_major and minor == minimum_minor and (revision >= minimum_revision))):
            return False
    else:
        return False
    return True