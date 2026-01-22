from __future__ import (absolute_import, division, print_function)
import json
import re
import sys
import datetime
import time
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
def fetch_all_resources(self, resource_key, **pagination_kwargs):
    resources = []
    result = [None]
    while len(result) != 0:
        result = self.fetch_paginated_resources(resource_key, **pagination_kwargs)
        resources += result
        if 'page' in pagination_kwargs:
            pagination_kwargs['page'] += 1
        else:
            pagination_kwargs['page'] = 2
    return resources