from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def update_tier_level(self, api_root, rest_api, headers, tier_level):
    body = {'level': tier_level}
    response, err, dummy = rest_api.post(api_root + 'change-tier-level', body, header=headers)
    if err is not None:
        return (False, 'Error: unexpected response on modify tier_level: %s, %s' % (str(err), str(response)))
    return (True, None)