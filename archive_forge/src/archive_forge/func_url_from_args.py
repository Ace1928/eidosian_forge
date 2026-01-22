from __future__ import (absolute_import, division, print_function)
import getpass
import os
import socket
import sys
import time
import uuid
from collections import OrderedDict
from os.path import basename
from ansible.errors import AnsibleError
from ansible.module_utils.six import raise_from
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.plugins.callback import CallbackBase
@staticmethod
def url_from_args(args):
    url_args = ('url', 'api_url', 'baseurl', 'repo', 'server_url', 'chart_repo_url', 'registry_url', 'endpoint', 'uri', 'updates_url')
    for arg in url_args:
        if args is not None and args.get(arg):
            return args.get(arg)
    return ''