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
def find_folder_by_fqpn(self, folder_name, datacenter_name=None, folder_type=None):
    """
        Get a unique folder managed object by specifying its Fully Qualified Path Name
        as datacenter/folder_type/sub1/sub2
        Args:
            folder_name: Fully Qualified Path Name folder name
            datacenter_name: Name of the datacenter, taken from Fully Qualified Path Name if not defined
            folder_type: Type of folder, vm, host, datastore or network,
                         taken from Fully Qualified Path Name if not defined

        Returns: folder managed object if found, else None

        """
    return find_folder_by_fqpn(self.content, folder_name=folder_name, datacenter_name=datacenter_name, folder_type=folder_type)