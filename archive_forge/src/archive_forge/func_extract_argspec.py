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
def extract_argspec(doc_obj, argpsec):
    options_obj = doc_obj.get('options')
    for okey, ovalue in iteritems(options_obj):
        argpsec[okey] = {}
        for metakey in list(ovalue):
            if metakey == 'suboptions':
                argpsec[okey].update({'options': {}})
                suboptions_obj = {'options': ovalue['suboptions']}
                extract_argspec(suboptions_obj, argpsec[okey]['options'])
            elif metakey in OPTION_METADATA + OPTION_CONDITIONALS:
                argpsec[okey].update({metakey: ovalue[metakey]})