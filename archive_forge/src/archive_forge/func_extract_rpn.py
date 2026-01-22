from __future__ import (absolute_import, division, print_function)
import json
from sys import version as python_version
from ansible.errors import AnsibleError
from ansible.module_utils.urls import open_url
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible.module_utils.six.moves.urllib.parse import urljoin
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def extract_rpn(self, host_infos):
    try:
        return self.rpn_lookup_cache[host_infos['id']]
    except (KeyError, TypeError):
        self.display.warning('An error happened while extracting RPN information. Information skipped.')
        return None