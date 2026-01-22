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
def get_all_hosts_by_cluster(self, cluster_name):
    """
        Get all hosts from cluster by cluster name
        Args:
            cluster_name: Name of cluster

        Returns: List of hosts

        """
    cluster_obj = self.find_cluster_by_name(cluster_name=cluster_name)
    if cluster_obj:
        return list(cluster_obj.host)
    else:
        return []