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
@staticmethod
def get_vm_path(content, vm_name):
    """
        Find the path of virtual machine.
        Args:
            content: VMware content object
            vm_name: virtual machine managed object

        Returns: Folder of virtual machine if exists, else None

        """
    folder_name = None
    folder = vm_name.parent
    if folder:
        folder_name = folder.name
        fp = folder.parent
        while fp is not None and fp.name is not None and (fp != content.rootFolder):
            folder_name = fp.name + '/' + folder_name
            try:
                fp = fp.parent
            except Exception:
                break
        folder_name = '/' + folder_name
    return folder_name