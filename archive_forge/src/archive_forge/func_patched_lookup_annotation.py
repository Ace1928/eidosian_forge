from __future__ import annotations
from functools import lru_cache
from typing import Any
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from docutils.parsers.rst.states import Text
from sphinx.application import Sphinx
from sphinx.ext.autodoc import Options
from sphinx.ext.napoleon.docstring import GoogleDocstring
from .attributes_patch import patch_attribute_handling
def patched_lookup_annotation(*_args: Any) -> str:
    """GoogleDocstring._lookup_annotation sometimes adds incorrect type
    annotations to constructor parameters (and otherwise does nothing). Disable
    it so we can handle this on our own.
    """
    return ''