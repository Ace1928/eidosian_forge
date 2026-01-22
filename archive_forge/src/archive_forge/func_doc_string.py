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
def doc_string(v):
    """Match a documentation string."""
    if not isinstance(v, string_types):
        raise _add_ansible_error_code(Invalid('Must be a string'), 'invalid-documentation')
    errors = []
    for par in parse(v, Context(), errors='message', strict=True, add_source=True):
        for part in par:
            if part.type == dom.PartType.ERROR:
                errors.append(_add_ansible_error_code(Invalid(part.message), 'invalid-documentation-markup'))
            if part.type == dom.PartType.URL:
                errors.extend(_check_url('U()', part.url))
            if part.type == dom.PartType.LINK:
                errors.extend(_check_url('L()', part.url))
            if part.type == dom.PartType.MODULE:
                if not FULLY_QUALIFIED_COLLECTION_RESOURCE_RE.match(part.fqcn):
                    errors.append(_add_ansible_error_code(Invalid('Directive "%s" must contain a FQCN; found "%s"' % (part.source, part.fqcn)), 'invalid-documentation-markup'))
            if part.type == dom.PartType.PLUGIN:
                if not FULLY_QUALIFIED_COLLECTION_RESOURCE_RE.match(part.plugin.fqcn):
                    errors.append(_add_ansible_error_code(Invalid('Directive "%s" must contain a FQCN; found "%s"' % (part.source, part.plugin.fqcn)), 'invalid-documentation-markup'))
                if part.plugin.type not in _VALID_PLUGIN_TYPES:
                    errors.append(_add_ansible_error_code(Invalid('Directive "%s" must contain a valid plugin type; found "%s"' % (part.source, part.plugin.type)), 'invalid-documentation-markup'))
            if part.type == dom.PartType.OPTION_NAME:
                if part.plugin is not None and (not FULLY_QUALIFIED_COLLECTION_RESOURCE_RE.match(part.plugin.fqcn)):
                    errors.append(_add_ansible_error_code(Invalid('Directive "%s" must contain a FQCN; found "%s"' % (part.source, part.plugin.fqcn)), 'invalid-documentation-markup'))
                if part.plugin is not None and part.plugin.type not in _VALID_PLUGIN_TYPES:
                    errors.append(_add_ansible_error_code(Invalid('Directive "%s" must contain a valid plugin type; found "%s"' % (part.source, part.plugin.type)), 'invalid-documentation-markup'))
            if part.type == dom.PartType.RETURN_VALUE:
                if part.plugin is not None and (not FULLY_QUALIFIED_COLLECTION_RESOURCE_RE.match(part.plugin.fqcn)):
                    errors.append(_add_ansible_error_code(Invalid('Directive "%s" must contain a FQCN; found "%s"' % (part.source, part.plugin.fqcn)), 'invalid-documentation-markup'))
                if part.plugin is not None and part.plugin.type not in _VALID_PLUGIN_TYPES:
                    errors.append(_add_ansible_error_code(Invalid('Directive "%s" must contain a valid plugin type; found "%s"' % (part.source, part.plugin.type)), 'invalid-documentation-markup'))
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise MultipleInvalid(errors)
    return v