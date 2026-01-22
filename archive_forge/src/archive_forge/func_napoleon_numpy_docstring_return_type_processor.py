from __future__ import annotations
from functools import lru_cache
from typing import Any
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from docutils.parsers.rst.states import Text
from sphinx.application import Sphinx
from sphinx.ext.autodoc import Options
from sphinx.ext.napoleon.docstring import GoogleDocstring
from .attributes_patch import patch_attribute_handling
def napoleon_numpy_docstring_return_type_processor(app: Sphinx, what: str, name: str, obj: Any, options: Options | None, lines: list[str]) -> None:
    """Insert a : under Returns: to tell napoleon not to look for a return type."""
    if what not in ['function', 'method']:
        return
    if not getattr(app.config, 'napoleon_numpy_docstring', False):
        return
    for idx, line in enumerate(lines[:-2]):
        if line.lower().strip(':') not in ['return', 'returns']:
            continue
        chars = set(lines[idx + 1].strip())
        if len(chars) != 1 or list(chars)[0] not in '=-~_*+#':
            continue
        idx = idx + 2
        break
    else:
        return
    lines.insert(idx, ':')