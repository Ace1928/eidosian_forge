from __future__ import (absolute_import, division, print_function)
import os
import time
import json
from ansible.utils.path import makedirs_safe
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.parsing.ajson import AnsibleJSONEncoder
from ansible.plugins.callback import CallbackBase

    logs playbook results, per host, in /var/log/ansible/hosts
    