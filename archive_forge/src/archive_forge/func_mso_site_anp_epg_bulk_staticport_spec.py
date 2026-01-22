from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import os
import ast
import datetime
import shutil
import tempfile
from ansible.module_utils.basic import json
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import filterfalse
from ansible.module_utils.six.moves.urllib.parse import urlencode, urljoin
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.mso.plugins.module_utils.constants import NDO_API_VERSION_PATH_FORMAT
def mso_site_anp_epg_bulk_staticport_spec():
    return dict(type=dict(type='str', choices=['port', 'vpc', 'dpc']), pod=dict(type='str'), leaf=dict(type='str'), fex=dict(type='str'), path=dict(type='str'), vlan=dict(type='int'), primary_micro_segment_vlan=dict(type='int'), deployment_immediacy=dict(type='str', choices=['immediate', 'lazy']), mode=dict(type='str', choices=['native', 'regular', 'untagged']))