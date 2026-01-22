from __future__ import absolute_import, division, print_function
import ast
import base64
import json
import os
import re
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.connection import ConnectionError
from ansible.plugins.httpapi import HttpApiBase
from copy import copy, deepcopy
def validate_url(self, url):
    validated_url = re.match('^.*?\\.json|^.*?\\.xml', url).group(0)
    if self.connection_parameters.get('port') is None:
        return validated_url.replace(re.match('(https?:\\/\\/.*)(:\\d*)\\/?(.*)', url).group(2), '')
    else:
        return validated_url