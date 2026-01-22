from __future__ import (absolute_import, division, print_function)
import logging
import json
import socket
from uuid import getnode
from ansible.plugins.callback import CallbackBase
from ansible.parsing.ajson import AnsibleJSONEncoder
def isJSONable(obj):
    try:
        json.dumps(obj, sort_keys=True, cls=AnsibleJSONEncoder)
        return True
    except Exception:
        return False