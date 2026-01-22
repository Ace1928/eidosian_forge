from __future__ import annotations
import datetime
import os
import re
import sys
from functools import partial
import yaml
from voluptuous import All, Any, MultipleInvalid, PREVENT_EXTRA
from voluptuous import Required, Schema, Invalid
from voluptuous.humanize import humanize_error
from ansible.module_utils.compat.version import StrictVersion, LooseVersion
from ansible.module_utils.six import string_types
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.version import SemanticVersion
def removal_version(value, is_ansible, current_version=None, is_tombstone=False):
    """Validate a removal version string."""
    msg = 'Removal version must be a string' if is_ansible else 'Removal version must be a semantic version (https://semver.org/)'
    if not isinstance(value, string_types):
        raise Invalid(msg)
    try:
        if is_ansible:
            version = StrictVersion()
            version.parse(value)
            version = LooseVersion(value)
        else:
            version = SemanticVersion()
            version.parse(value)
            if version.major != 0 and (version.minor != 0 or version.patch != 0):
                raise Invalid('removal_version (%r) must be a major release, not a minor or patch release (see specification at https://semver.org/)' % (value,))
        if current_version is not None:
            if is_tombstone:
                if version > current_version:
                    raise Invalid('The tombstone removal_version (%r) must not be after the current version (%s)' % (value, current_version))
            elif version <= current_version:
                raise Invalid('The deprecation removal_version (%r) must be after the current version (%s)' % (value, current_version))
    except ValueError:
        raise Invalid(msg)
    return value