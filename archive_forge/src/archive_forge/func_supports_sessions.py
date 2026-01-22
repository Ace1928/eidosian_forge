from __future__ import absolute_import, division, print_function
import json
import os
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
@property
def supports_sessions(self):
    if self._session_support is None:
        self._session_support = self._connection.supports_sessions()
    return self._session_support