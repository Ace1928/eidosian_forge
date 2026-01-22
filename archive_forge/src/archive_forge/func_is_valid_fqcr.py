from __future__ import (absolute_import, division, print_function)
import itertools
import os
import os.path
import pkgutil
import re
import sys
from keyword import iskeyword
from tokenize import Name as _VALID_IDENTIFIER_REGEX
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.six import string_types, PY3
from ._collection_config import AnsibleCollectionConfig
from contextlib import contextmanager
from types import ModuleType
@staticmethod
def is_valid_fqcr(ref, ref_type=None):
    """
        Validates if is string is a well-formed fully-qualified collection reference (does not look up the collection itself)
        :param ref: candidate collection reference to validate (a valid ref is of the form 'ns.coll.resource' or 'ns.coll.subdir1.subdir2.resource')
        :param ref_type: optional reference type to enable deeper validation, eg 'module', 'role', 'doc_fragment'
        :return: True if the collection ref passed is well-formed, False otherwise
        """
    ref = to_text(ref)
    if not ref_type:
        return bool(re.match(AnsibleCollectionRef.VALID_FQCR_RE, ref))
    return bool(AnsibleCollectionRef.try_parse_fqcr(ref, ref_type))