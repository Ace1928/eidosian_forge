import os
import posixpath
import re
import urllib.parse
import warnings
from typing import TYPE_CHECKING, Iterable, Optional, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node, Text
from docutils.writers.html4css1 import HTMLTranslator as BaseTranslator
from docutils.writers.html4css1 import Writer
from sphinx import addnodes
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.locale import _, __, admonitionlabels
from sphinx.util import logging
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.images import get_image_size
def get_secnumber(self, node: Element) -> Optional[Tuple[int, ...]]:
    if node.get('secnumber'):
        return node['secnumber']
    elif isinstance(node.parent, nodes.section):
        if self.builder.name == 'singlehtml':
            docname = self.docnames[-1]
            anchorname = '%s/#%s' % (docname, node.parent['ids'][0])
            if anchorname not in self.builder.secnumbers:
                anchorname = '%s/' % docname
        else:
            anchorname = '#' + node.parent['ids'][0]
            if anchorname not in self.builder.secnumbers:
                anchorname = ''
        if self.builder.secnumbers.get(anchorname):
            return self.builder.secnumbers[anchorname]
    return None