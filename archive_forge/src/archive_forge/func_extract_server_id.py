from __future__ import (absolute_import, division, print_function)
import os
import json
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, parse_pagination_link
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.six import raise_from
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
import ansible.module_utils.six.moves.urllib.parse as urllib_parse
def extract_server_id(server_info):
    try:
        return server_info['id']
    except (KeyError, TypeError):
        return None