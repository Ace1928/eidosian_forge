from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import os
import ast
import datetime
import shutil
import tempfile
from ansible.module_utils.basic import json
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import filterfalse
from ansible.module_utils.six.moves.urllib.parse import urlencode, urljoin
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.mso.plugins.module_utils.constants import NDO_API_VERSION_PATH_FORMAT
def remove_keys_from_dict_when_value_empty(self, target_dict, modified_target=None):
    if modified_target is None:
        modified_target = deepcopy(target_dict)
    for key, value in target_dict.items():
        if value is None:
            del modified_target[key]
        elif isinstance(value, dict):
            self.remove_keys_from_dict_when_value_empty(value, modified_target[key])
        elif isinstance(value, list):
            for entry_index, entry in enumerate(value):
                if isinstance(entry, dict):
                    self.remove_keys_from_dict_when_value_empty(entry, modified_target[key][entry_index])
    return modified_target