from inspect import signature, Signature
from typing import (
import ast
import builtins
import collections
import operator
import sys
from functools import cached_property
from dataclasses import dataclass, field
from types import MethodDescriptorType, ModuleType
from IPython.utils.docs import GENERATING_DOCUMENTATION
from IPython.utils.decorators import undoc
def can_get_attr(self, value, attr):
    has_original_attribute = _has_original_dunder(value, allowed_types=self.allowed_getattr, allowed_methods=self._getattribute_methods, allowed_external=self.allowed_getattr_external, method_name='__getattribute__')
    has_original_attr = _has_original_dunder(value, allowed_types=self.allowed_getattr, allowed_methods=self._getattr_methods, allowed_external=self.allowed_getattr_external, method_name='__getattr__')
    accept = False
    if has_original_attr is None and has_original_attribute:
        accept = True
    else:
        accept = has_original_attr and has_original_attribute
    if accept:
        value_class = type(value)
        if not hasattr(value_class, attr):
            return True
        class_attr_val = getattr(value_class, attr)
        is_property = isinstance(class_attr_val, property)
        if not is_property:
            return True
        if type(value) in self.allowed_getattr:
            return True
        for module_name, *access_path in self.allowed_getattr_external:
            try:
                external_class = _get_external(module_name, access_path)
                external_class_attr_val = getattr(external_class, attr)
            except (KeyError, AttributeError):
                return False
            return class_attr_val == external_class_attr_val
    return False