from __future__ import (absolute_import, division, print_function)
import os
import shutil
import traceback
import select
import fcntl
import errno
from ansible import errors
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.plugins.connection import ConnectionBase
 terminate the connection; nothing to do here 