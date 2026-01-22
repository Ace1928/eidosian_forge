import html
import os
import re
from os import path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple
from urllib.parse import quote
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.utils import smartquotes
from sphinx import addnodes
from sphinx.builders.html import BuildInfo, StandaloneHTMLBuilder
from sphinx.locale import __
from sphinx.util import logging, status_iterator
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.i18n import format_date
from sphinx.util.osutil import copyfile, ensuredir
def footnote_spot(tree: nodes.document) -> Tuple[Element, int]:
    """Find or create a spot to place footnotes.

            The function returns the tuple (parent, index)."""
    fns = list(tree.findall(nodes.footnote))
    if fns:
        fn = fns[-1]
        return (fn.parent, fn.parent.index(fn) + 1)
    for node in tree.findall(nodes.rubric):
        if len(node) == 1 and node.astext() == FOOTNOTES_RUBRIC_NAME:
            return (node.parent, node.parent.index(node) + 1)
    doc = next(tree.findall(nodes.document))
    rub = nodes.rubric()
    rub.append(nodes.Text(FOOTNOTES_RUBRIC_NAME))
    doc.append(rub)
    return (doc, doc.index(rub) + 1)