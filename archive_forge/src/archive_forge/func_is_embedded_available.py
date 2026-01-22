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
def is_embedded_available(self):
    """Determine whether the storage array has embedded services available."""
    self._check_web_services_version()
    if self.is_embedded_available_cache is None:
        if self.is_proxy():
            if self.ssid == '0' or self.ssid.lower() == 'proxy':
                self.is_embedded_available_cache = False
            else:
                try:
                    rc, bundle = self.request("storage-systems/%s/graph/xpath-filter?query=/sa/saData/extendedSAData/codeVersions[codeModule='bundle']" % self.ssid)
                    self.is_embedded_available_cache = False
                    if bundle:
                        self.is_embedded_available_cache = True
                except Exception as error:
                    self.module.fail_json(msg='Failed to retrieve information about storage system [%s]. Error [%s].' % (self.ssid, to_native(error)))
        else:
            self.is_embedded_available_cache = True
        self.module.log('embedded_available: [%s]' % ('True' if self.is_embedded_available_cache else 'False'))
    return self.is_embedded_available_cache