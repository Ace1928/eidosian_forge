from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
@staticmethod
def get_logging_from_ansible_params(params):
    logging_config = params['logging'] or {}
    driver = get_value('driver', logging_config)
    options = get_value('options', logging_config)
    return {'log_driver': driver, 'log_driver_options': options}