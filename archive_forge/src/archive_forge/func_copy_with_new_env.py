from __future__ import (absolute_import, division, print_function)
import ast
import datetime
import os
import pwd
import re
import time
from collections.abc import Iterator, Sequence, Mapping, MappingView, MutableMapping
from contextlib import contextmanager
from numbers import Number
from traceback import format_exc
from jinja2.exceptions import TemplateSyntaxError, UndefinedError, SecurityError
from jinja2.loaders import FileSystemLoader
from jinja2.nativetypes import NativeEnvironment
from jinja2.runtime import Context, StrictUndefined
from ansible import constants as C
from ansible.errors import (
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.common.collections import is_sequence
from ansible.plugins.loader import filter_loader, lookup_loader, test_loader
from ansible.template.native_helpers import ansible_native_concat, ansible_eval_concat, ansible_concat
from ansible.template.template import AnsibleJ2Template
from ansible.template.vars import AnsibleJ2Vars
from ansible.utils.display import Display
from ansible.utils.listify import listify_lookup_plugin_terms
from ansible.utils.native_jinja import NativeJinjaText
from ansible.utils.unsafe_proxy import to_unsafe_text, wrap_var, AnsibleUnsafeText, AnsibleUnsafeBytes, NativeJinjaUnsafeText
def copy_with_new_env(self, environment_class=AnsibleEnvironment, **kwargs):
    """Creates a new copy of Templar with a new environment.

        :kwarg environment_class: Environment class used for creating a new environment.
        :kwarg \\*\\*kwargs: Optional arguments for the new environment that override existing
            environment attributes.

        :returns: Copy of Templar with updated environment.
        """
    new_env = object.__new__(environment_class)
    new_env.__dict__.update(self.environment.__dict__)
    new_templar = object.__new__(Templar)
    new_templar.__dict__.update(self.__dict__)
    new_templar.environment = new_env
    new_templar.jinja2_native = environment_class is AnsibleNativeEnvironment
    mapping = {'available_variables': new_templar, 'searchpath': new_env.loader}
    for key, value in kwargs.items():
        obj = mapping.get(key, new_env)
        try:
            if value is not None:
                setattr(obj, key, value)
        except AttributeError:
            pass
    return new_templar