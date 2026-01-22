from __future__ import annotations
import html
import os
from os import path
from typing import Any
from docutils import nodes
from docutils.nodes import Element, Node, document
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.config import Config
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.nodes import NodeMatcher
from sphinx.util.osutil import make_filename_from_project, relpath
from sphinx.util.template import SphinxRenderer
@progress_message(__('copying stopword list'))
def copy_stopword_list(self) -> None:
    """Copy a stopword list (.stp) to outdir.

        The stopword list contains a list of words the full text search facility
        shouldn't index.  Note that this list must be pretty small.  Different
        versions of the MS docs claim the file has a maximum size of 256 or 512
        bytes (including \r
 at the end of each line).  Note that "and", "or",
        "not" and "near" are operators in the search language, so no point
        indexing them even if we wanted to.
        """
    template = path.join(template_dir, 'project.stp')
    filename = path.join(self.outdir, self.config.htmlhelp_basename + '.stp')
    copy_asset_file(template, filename)