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
def find_first_class_disks(self, datastore_obj):
    """
        Get first-class disks managed object
        Args:
            datastore_obj: Managed object of datastore

        Returns: First-class disks managed object if found else None

        """
    disks = []
    if self.is_vcenter():
        for id in self.content.vStorageObjectManager.ListVStorageObject(datastore_obj):
            disks.append(self.content.vStorageObjectManager.RetrieveVStorageObject(id, datastore_obj))
    else:
        for id in self.content.vStorageObjectManager.HostListVStorageObject(datastore_obj):
            disks.append(self.content.vStorageObjectManager.HostRetrieveVStorageObject(id, datastore_obj))
    if disks == []:
        return None
    else:
        return disks