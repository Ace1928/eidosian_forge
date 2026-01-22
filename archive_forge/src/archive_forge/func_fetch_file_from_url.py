from __future__ import absolute_import, division, print_function
import base64
import time
import os
import traceback
import sys
import hashlib
from datetime import datetime
from tempfile import NamedTemporaryFile
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.hashes import (
from ansible_collections.kubernetes.core.plugins.module_utils.selector import (
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems, string_types
from ansible.module_utils._text import to_native, to_bytes, to_text
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.urls import Request
def fetch_file_from_url(module, url):
    bufsize = 65536
    file_name, file_ext = os.path.splitext(str(url.rsplit('/', 1)[1]))
    temp_file = NamedTemporaryFile(dir=module.tmpdir, prefix=file_name, suffix=file_ext, delete=False)
    module.add_cleanup_file(temp_file.name)
    try:
        rsp = Request().open('GET', url)
        if not rsp:
            module.fail_json(msg='Failure downloading %s' % url)
        data = rsp.read(bufsize)
        while data:
            temp_file.write(data)
            data = rsp.read(bufsize)
        temp_file.close()
    except Exception as e:
        module.fail_json(msg='Failure downloading %s, %s' % (url, to_native(e)))
    return temp_file.name