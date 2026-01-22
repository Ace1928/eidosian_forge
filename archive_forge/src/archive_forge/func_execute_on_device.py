from __future__ import absolute_import, division, print_function
import os
import re
import socket
import ssl
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def execute_on_device(self):
    self._upsert_temporary_cli_script_on_device()
    task_id = self._create_async_task_on_device()
    self._exec_async_task_on_device(task_id)
    self._wait_for_async_task_to_finish_on_device(task_id)
    self._remove_temporary_cli_script_from_device()
    return True