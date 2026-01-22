from __future__ import absolute_import, division, print_function
import os
import re
import shutil
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlsplit
from ansible.module_utils.compat.datetime import utcnow, utcfromtimestamp
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import fetch_url, url_argument_spec
def extract_filename_from_headers(headers):
    """
    Extracts a filename from the given dict of HTTP headers.

    Looks for the content-disposition header and applies a regex.
    Returns the filename if successful, else None."""
    cont_disp_regex = 'attachment; ?filename="?([^"]+)'
    res = None
    if 'content-disposition' in headers:
        cont_disp = headers['content-disposition']
        match = re.match(cont_disp_regex, cont_disp)
        if match:
            res = match.group(1)
            res = os.path.basename(res)
    return res