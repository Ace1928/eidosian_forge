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
def rst_to_docutils(settings: Values, rst: str) -> Any:
    """Convert rst to a sequence of docutils nodes"""
    doc = new_document('', settings)
    RstParser().parse(rst, doc)
    return doc.children[0].children