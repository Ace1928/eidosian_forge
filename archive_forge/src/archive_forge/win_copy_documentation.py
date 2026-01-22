from __future__ import (absolute_import, division, print_function)
import base64
import json
import os
import os.path
import shutil
import tempfile
import traceback
import zipfile
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleFileNotFound
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
from ansible.utils.hashing import checksum
 handler for file transfer operations 