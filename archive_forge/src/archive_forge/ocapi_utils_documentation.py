from __future__ import absolute_import, division, print_function
import json
import os
import uuid
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
Delete the OCAPI job referenced by the specified job_uri.