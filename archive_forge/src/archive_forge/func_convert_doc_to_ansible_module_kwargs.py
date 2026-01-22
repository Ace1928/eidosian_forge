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
def convert_doc_to_ansible_module_kwargs(doc):
    doc_obj = yaml.load(str(doc), SafeLoader)
    argspec = {}
    spec = {}
    extract_argspec(doc_obj, argspec)
    spec.update({'argument_spec': argspec})
    for item in doc_obj:
        if item in VALID_ANSIBLEMODULE_ARGS:
            spec.update({item: doc_obj[item]})
    return spec