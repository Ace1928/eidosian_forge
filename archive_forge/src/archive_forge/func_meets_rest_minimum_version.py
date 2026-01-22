from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def meets_rest_minimum_version(self, use_rest, minimum_generation, minimum_major, minimum_minor=0):
    return use_rest and self.get_ontap_version() >= (minimum_generation, minimum_major, minimum_minor)