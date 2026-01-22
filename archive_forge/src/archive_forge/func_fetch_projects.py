from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ..module_utils.gcp_utils import (
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
def fetch_projects(self, params, link, query):
    module = GcpMockModule(params)
    auth = GcpSession(module, 'cloudresourcemanager')
    response = auth.get(link, params={'filter': query})
    return self._return_if_object(module, response)