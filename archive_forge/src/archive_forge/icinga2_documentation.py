from __future__ import (absolute_import, division, print_function)
import json
from ansible.errors import AnsibleParserError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
Convert Icinga2 API data to JSON format for Ansible