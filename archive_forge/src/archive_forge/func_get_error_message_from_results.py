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
def get_error_message_from_results(results, action):
    for result in results:
        if result.get('failed', False):
            return '{0}({1}) - {2}'.format(action, result.get('item', 'none'), OpenTelemetrySource.get_error_message(result))