from __future__ import (absolute_import, division, print_function)
import json
import ssl
from time import sleep
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
Calls a method on the XO server with the provided parameters.