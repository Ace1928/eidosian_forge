from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def find_object_by_name(content, name, obj_type, folder=None, recurse=True):
    if not isinstance(obj_type, list):
        obj_type = [obj_type]
    name = name.strip()
    objects = get_all_objs(content, obj_type, folder=folder, recurse=recurse)
    for obj in objects:
        try:
            if unquote(obj.name) == name:
                return obj
        except vmodl.fault.ManagedObjectNotFound:
            pass
    return None