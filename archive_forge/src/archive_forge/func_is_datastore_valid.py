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
def is_datastore_valid(self, datastore_obj=None):
    """
        Check if datastore selected is valid or not
        Args:
            datastore_obj: datastore managed object

        Returns: True if datastore is valid, False if not
        """
    if not datastore_obj or datastore_obj.summary.maintenanceMode != 'normal' or (not datastore_obj.summary.accessible):
        return False
    return True