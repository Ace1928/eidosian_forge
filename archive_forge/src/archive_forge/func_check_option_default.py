from __future__ import annotations
import re
from ansible.module_utils.compat.version import StrictVersion
from functools import partial
from urllib.parse import urlparse
from voluptuous import ALLOW_EXTRA, PREVENT_EXTRA, All, Any, Invalid, Length, MultipleInvalid, Required, Schema, Self, ValueInvalid, Exclusive
from ansible.constants import DOCUMENTABLE_PLUGINS
from ansible.module_utils.six import string_types
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.parsing.quoting import unquote
from ansible.utils.version import SemanticVersion
from ansible.release import __version__
from antsibull_docs_parser import dom
from antsibull_docs_parser.parser import parse, Context
from .utils import parse_isodate
def check_option_default(v):
    v_default = v.get('default')
    if v.get('required') and v_default is not None:
        raise _add_ansible_error_code(Invalid('Argument is marked as required but specifies a default. Arguments with a default should not be marked as required'), error_code='no-default-for-required-parameter')
    if v_default is None:
        return v
    type_checker, type_name = get_type_checker(v)
    if type_checker is None:
        return v
    try:
        type_checker(v_default)
    except Exception as exc:
        raise _add_ansible_error_code(Invalid('Argument defines default as (%r) but this is incompatible with parameter type %s: %s' % (v_default, type_name, exc)), error_code='incompatible-default-type')
    return v