from __future__ import absolute_import, division, print_function
from awx.main.tests.functional.conftest import _request
from ansible.module_utils.six import string_types
import yaml
import os
import re
import glob
def cause_error(msg):
    global return_value
    return_value = 255
    return msg