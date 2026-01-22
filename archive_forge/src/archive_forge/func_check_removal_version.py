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
def check_removal_version(v, version_field, collection_name_field, error_code='invalid-removal-version'):
    version = v.get(version_field)
    collection_name = v.get(collection_name_field)
    if not isinstance(version, string_types) or not isinstance(collection_name, string_types):
        return v
    if collection_name == 'ansible.builtin':
        try:
            parsed_version = StrictVersion()
            parsed_version.parse(version)
        except ValueError as exc:
            raise _add_ansible_error_code(Invalid('%s (%r) is not a valid ansible-core version: %s' % (version_field, version, exc)), error_code=error_code)
        return v
    try:
        parsed_version = SemanticVersion()
        parsed_version.parse(version)
        if parsed_version.major != 0 and (parsed_version.minor != 0 or parsed_version.patch != 0):
            raise _add_ansible_error_code(Invalid('%s (%r) must be a major release, not a minor or patch release (see specification at https://semver.org/)' % (version_field, version)), error_code='removal-version-must-be-major')
    except ValueError as exc:
        raise _add_ansible_error_code(Invalid('%s (%r) is not a valid collection version (see specification at https://semver.org/): %s' % (version_field, version, exc)), error_code=error_code)
    return v