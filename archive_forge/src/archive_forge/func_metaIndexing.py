from __future__ import (absolute_import, division, print_function)
import logging
import json
import socket
from uuid import getnode
from ansible.plugins.callback import CallbackBase
from ansible.parsing.ajson import AnsibleJSONEncoder
def metaIndexing(self, meta):
    invalidKeys = []
    ninvalidKeys = 0
    for key, value in meta.items():
        if not isJSONable(value):
            invalidKeys.append(key)
            ninvalidKeys += 1
    if ninvalidKeys > 0:
        for key in invalidKeys:
            del meta[key]
        meta['__errors'] = 'These keys have been sanitized: ' + ', '.join(invalidKeys)
    return meta