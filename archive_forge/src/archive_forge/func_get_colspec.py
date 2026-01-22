import re
import warnings
from collections import defaultdict
from os import path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, cast
from docutils import nodes, writers
from docutils.nodes import Element, Node, Text
from sphinx import addnodes, highlighting
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.domains import IndexEntry
from sphinx.domains.std import StandardDomain
from sphinx.errors import SphinxError
from sphinx.locale import _, __, admonitionlabels
from sphinx.util import logging, split_into, texescape
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.nodes import clean_astext, get_prev_node
from sphinx.util.template import LaTeXRenderer
from sphinx.util.texescape import tex_replace_map
from sphinx.builders.latex.nodes import ( # NOQA isort:skip
def get_colspec(self) -> str:
    """Returns a column spec of table.

        This is what LaTeX calls the 'preamble argument' of the used table environment.

        .. note::

           The ``\\X`` and ``T`` column type specifiers are defined in
           ``sphinxlatextables.sty``.
        """
    if self.colspec:
        return self.colspec
    _colsep = self.colsep
    if self.colwidths and 'colwidths-given' in self.classes:
        total = sum(self.colwidths)
        colspecs = ['\\X{%d}{%d}' % (width, total) for width in self.colwidths]
        return '{%s%s%s}' % (_colsep, _colsep.join(colspecs), _colsep) + CR
    elif self.has_problematic:
        return '{%s*{%d}{\\X{1}{%d}%s}}' % (_colsep, self.colcount, self.colcount, _colsep) + CR
    elif self.get_table_type() == 'tabulary':
        return '{' + _colsep + ('T' + _colsep) * self.colcount + '}' + CR
    elif self.has_oldproblematic:
        return '{%s*{%d}{\\X{1}{%d}%s}}' % (_colsep, self.colcount, self.colcount, _colsep) + CR
    else:
        return '{' + _colsep + ('l' + _colsep) * self.colcount + '}' + CR