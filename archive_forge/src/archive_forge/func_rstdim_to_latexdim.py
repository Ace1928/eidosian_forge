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
def rstdim_to_latexdim(width_str: str, scale: int=100) -> str:
    """Convert `width_str` with rst length to LaTeX length."""
    match = re.match('^(\\d*\\.?\\d*)\\s*(\\S*)$', width_str)
    if not match:
        raise ValueError
    res = width_str
    amount, unit = match.groups()[:2]
    if scale == 100:
        float(amount)
        if unit in ('', 'px'):
            res = '%s\\sphinxpxdimen' % amount
        elif unit == 'pt':
            res = '%sbp' % amount
        elif unit == '%':
            res = '%.3f\\linewidth' % (float(amount) / 100.0)
    else:
        amount_float = float(amount) * scale / 100.0
        if unit in ('', 'px'):
            res = '%.5f\\sphinxpxdimen' % amount_float
        elif unit == 'pt':
            res = '%.5fbp' % amount_float
        elif unit == '%':
            res = '%.5f\\linewidth' % (amount_float / 100.0)
        else:
            res = '%.5f%s' % (amount_float, unit)
    return res