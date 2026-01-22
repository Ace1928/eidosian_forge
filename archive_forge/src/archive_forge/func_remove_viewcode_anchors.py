import posixpath
import traceback
from os import path
from typing import Any, Dict, Generator, Iterable, Optional, Set, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import get_full_modname, logging, status_iterator
from sphinx.util.nodes import make_refnode
def remove_viewcode_anchors(self) -> None:
    for node in list(self.document.findall(viewcode_anchor)):
        node.parent.remove(node)