from __future__ import annotations
import ast
import re
import typing as t
from dataclasses import dataclass
from string import Template
from types import CodeType
from urllib.parse import quote
from ..datastructures import iter_multi_items
from ..urls import _urlencode
from .converters import ValidationError
def get_empty_kwargs(self) -> t.Mapping[str, t.Any]:
    """
        Provides kwargs for instantiating empty copy with empty()

        Use this method to provide custom keyword arguments to the subclass of
        ``Rule`` when calling ``some_rule.empty()``.  Helpful when the subclass
        has custom keyword arguments that are needed at instantiation.

        Must return a ``dict`` that will be provided as kwargs to the new
        instance of ``Rule``, following the initial ``self.rule`` value which
        is always provided as the first, required positional argument.
        """
    defaults = None
    if self.defaults:
        defaults = dict(self.defaults)
    return dict(defaults=defaults, subdomain=self.subdomain, methods=self.methods, build_only=self.build_only, endpoint=self.endpoint, strict_slashes=self.strict_slashes, redirect_to=self.redirect_to, alias=self.alias, host=self.host)