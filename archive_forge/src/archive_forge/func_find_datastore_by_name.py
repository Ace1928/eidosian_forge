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
def find_datastore_by_name(self, datastore_name, datacenter_name=None):
    """
        Get datastore managed object by name
        Args:
            datastore_name: Name of datastore
            datacenter_name: Name of datacenter where the datastore resides.  This is needed because Datastores can be
            shared across Datacenters, so we need to specify the datacenter to assure we get the correct Managed Object Reference

        Returns: datastore managed object if found else None

        """
    return find_datastore_by_name(self.content, datastore_name=datastore_name, datacenter_name=datacenter_name)