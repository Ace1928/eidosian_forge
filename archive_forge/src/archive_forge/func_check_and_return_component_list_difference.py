from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def check_and_return_component_list_difference(input_component_list, existing_components, purge_components, delete_components=False):
    if input_component_list:
        existing_components, changed = get_component_list_difference(input_component_list, existing_components, purge_components, delete_components)
    else:
        existing_components = []
        changed = True
    return (existing_components, changed)