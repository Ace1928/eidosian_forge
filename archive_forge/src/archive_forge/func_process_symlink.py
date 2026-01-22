from __future__ import (absolute_import, division, print_function)
import base64
import datetime
import io
import json
import os
import os.path
import shutil
import stat
import tarfile
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import raise_from
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, NotFound
def process_symlink(in_path, member):
    if os.path.exists(b_out_path):
        os.unlink(b_out_path)
    os.symlink(member.linkname, b_out_path)
    return in_path