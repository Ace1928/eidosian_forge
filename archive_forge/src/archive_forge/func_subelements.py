from __future__ import (absolute_import, division, print_function)
import base64
import glob
import hashlib
import json
import ntpath
import os.path
import re
import shlex
import sys
import time
import uuid
import yaml
import datetime
from collections.abc import Mapping
from functools import partial
from random import Random, SystemRandom, shuffle
from jinja2.filters import pass_environment
from ansible.errors import AnsibleError, AnsibleFilterError, AnsibleFilterTypeError
from ansible.module_utils.six import string_types, integer_types, reraise, text_type
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.yaml import yaml_load, yaml_load_all
from ansible.parsing.ajson import AnsibleJSONEncoder
from ansible.parsing.yaml.dumper import AnsibleDumper
from ansible.template import recursive_check_defined
from ansible.utils.display import Display
from ansible.utils.encrypt import do_encrypt, PASSLIB_AVAILABLE
from ansible.utils.hashing import md5s, checksum_s
from ansible.utils.unicode import unicode_wrap
from ansible.utils.unsafe_proxy import _is_unsafe
from ansible.utils.vars import merge_hash
def subelements(obj, subelements, skip_missing=False):
    """Accepts a dict or list of dicts, and a dotted accessor and produces a product
    of the element and the results of the dotted accessor

    >>> obj = [{"name": "alice", "groups": ["wheel"], "authorized": ["/tmp/alice/onekey.pub"]}]
    >>> subelements(obj, 'groups')
    [({'name': 'alice', 'groups': ['wheel'], 'authorized': ['/tmp/alice/onekey.pub']}, 'wheel')]

    """
    if isinstance(obj, dict):
        element_list = list(obj.values())
    elif isinstance(obj, list):
        element_list = obj[:]
    else:
        raise AnsibleFilterError('obj must be a list of dicts or a nested dict')
    if isinstance(subelements, list):
        subelement_list = subelements[:]
    elif isinstance(subelements, string_types):
        subelement_list = subelements.split('.')
    else:
        raise AnsibleFilterTypeError('subelements must be a list or a string')
    results = []
    for element in element_list:
        values = element
        for subelement in subelement_list:
            try:
                values = values[subelement]
            except KeyError:
                if skip_missing:
                    values = []
                    break
                raise AnsibleFilterError('could not find %r key in iterated item %r' % (subelement, values))
            except TypeError:
                raise AnsibleFilterTypeError("the key %s should point to a dictionary, got '%s'" % (subelement, values))
        if not isinstance(values, list):
            raise AnsibleFilterTypeError('the key %r should point to a list, got %r' % (subelement, values))
        for value in values:
            results.append((element, value))
    return results