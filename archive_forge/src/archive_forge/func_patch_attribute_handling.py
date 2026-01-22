from functools import partial
from optparse import Values
from typing import Any, Tuple
from unittest.mock import patch
import sphinx.domains.python
import sphinx.ext.autodoc
from docutils.parsers.rst import Parser as RstParser
from docutils.utils import new_document
from sphinx.addnodes import desc_signature
from sphinx.application import Sphinx
from sphinx.domains.python import PyAttribute
from sphinx.ext.autodoc import AttributeDocumenter
def patch_attribute_handling(app: Sphinx) -> None:
    """Use format_signature to format class attribute type annotations"""
    if not OKAY_TO_PATCH:
        return
    PyAttribute.handle_signature = patched_handle_signature
    patch_attribute_documenter(app)