from __future__ import absolute_import, division, print_function
import ast
import json
import operator
import re
import socket
from copy import deepcopy
from functools import reduce  # forward compatibility for Python 3
from itertools import chain
from ansible.module_utils import basic
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems, string_types
def parse_conf_arg(cfg, arg):
    """
    Parse config based on argument

    :param cfg: A text string which is a line of configuration.
    :param arg: A text string which is to be matched.
    :rtype: A text string
    :returns: A text string if match is found
    """
    match = re.search('%s (.+)(\\n|$)' % arg, cfg, re.M)
    if match:
        result = match.group(1).strip()
    else:
        result = None
    return result